const fs = require("fs");
const express = require("express");
const bodyParser = require("body-parser");
const mineflayer = require("mineflayer");

const skills = require("./lib/skillLoader");
const {initCounter, getNextTime} = require("./lib/utils");
const obs = require("./lib/observation/base");
const OnChat = require("./lib/observation/onChat");
const OnError = require("./lib/observation/onError");
const {Voxels, BlockRecords} = require("./lib/observation/voxels");
const Status = require("./lib/observation/status");
const Inventory = require("./lib/observation/inventory");
const OnSave = require("./lib/observation/onSave");
const Chests = require("./lib/observation/chests");
const {plugin: tool} = require("mineflayer-tool");

const {pathfinder, Movements, goals} = require('mineflayer-pathfinder')
const {GoalXZ, GoalBlock} = goals

let bot = null;
let viewerStatus = "disabled";
let viewerHost = "127.0.0.1";

function isAirLike(block) {
    return !block || block.name === "air" || block.name === "cave_air" || block.name === "void_air";
}

function isStandableGround(block) {
    return block && block.name !== "air" && block.boundingBox === "block";
}

function hasPlaceableAdjacentSpot(botInstance, standPos) {
    const candidates = [
        {x: 1, z: 0},
        {x: -1, z: 0},
        {x: 0, z: 1},
        {x: 0, z: -1},
        {x: 1, z: 1},
        {x: -1, z: 1},
        {x: 1, z: -1},
        {x: -1, z: -1},
    ];
    for (const offset of candidates) {
        const ground = botInstance.blockAt(standPos.offset(offset.x, -1, offset.z));
        const place = botInstance.blockAt(standPos.offset(offset.x, 0, offset.z));
        const head = botInstance.blockAt(standPos.offset(offset.x, 1, offset.z));
        if (isStandableGround(ground) && isAirLike(place) && isAirLike(head)) {
            return true;
        }
    }
    return false;
}

function findSafeTrackedPosition(botInstance, playerEntity, searchRadius = 6) {
    const base = playerEntity.position.floored();
    const offsets = [];
    for (let radius = 2; radius <= searchRadius; radius++) {
        for (let dx = -radius; dx <= radius; dx++) {
            for (let dz = -radius; dz <= radius; dz++) {
                if (Math.max(Math.abs(dx), Math.abs(dz)) !== radius) continue;
                offsets.push({dx, dz});
            }
        }
    }

    for (const {dx, dz} of offsets) {
        for (const dy of [0, -1, 1, -2, 2]) {
            const feetPos = base.offset(dx, dy, dz);
            const ground = botInstance.blockAt(feetPos.offset(0, -1, 0));
            const feet = botInstance.blockAt(feetPos);
            const head = botInstance.blockAt(feetPos.offset(0, 1, 0));
            if (!isStandableGround(ground)) continue;
            if (!isAirLike(feet) || !isAirLike(head)) continue;
            if (!hasPlaceableAdjacentSpot(botInstance, feetPos)) continue;
            return {
                x: feetPos.x + 0.5,
                y: feetPos.y,
                z: feetPos.z + 0.5,
            };
        }
    }

    return null;
}

function getNearestHumanPlayerEntity(botInstance) {
    const candidates = Object.values(botInstance.players || {})
        .map((player) => player && player.entity ? player.entity : null)
        .filter((entity) => entity && entity.username && entity.username !== botInstance.username);

    if (!candidates.length) {
        return null;
    }

    return candidates.sort((a, b) => {
        return a.position.distanceTo(botInstance.entity.position) - b.position.distanceTo(botInstance.entity.position);
    })[0];
}

async function runCommand(botInstance, command, extraTicks = 1) {
    botInstance.chat(command);
    await botInstance.waitForTicks(botInstance.waitTicks * extraTicks);
}

function getTrackedPlayerSpawnPosition(playerEntity) {
    const baseX = playerEntity.position.x;
    const baseY = playerEntity.position.y;
    const baseZ = playerEntity.position.z;
    const offsetDistance = 4;
    const sideOffset = 1.5;
    const yaw = playerEntity.yaw || 0;
    const behindX = -Math.sin(yaw) * offsetDistance;
    const behindZ = Math.cos(yaw) * offsetDistance;
    const sideX = Math.cos(yaw) * sideOffset;
    const sideZ = Math.sin(yaw) * sideOffset;

    return {
        x: (baseX + behindX + sideX).toFixed(2),
        y: Math.floor(baseY),
        z: (baseZ + behindZ + sideZ).toFixed(2),
    };
}

const app = express();

app.use(bodyParser.json({limit: "50mb"}));
app.use(bodyParser.urlencoded({limit: "50mb", extended: false}));

app.post("/start", (req, res) => {
    if (bot) onDisconnect("Restarting bot");
    bot = null;
    viewerStatus = "disabled";
    console.log(req.body);
    console.log(`Connecting mineflayer bot to localhost:${req.body.port} as bot_${PORT}`);
    bot = mineflayer.createBot({
        host: "localhost", // minecraft server ip
        port: req.body.port, // minecraft server port
        username: `bot_${PORT}`,
        auth: "offline",
        version: "1.19",
        disableChatSigning: true,
        checkTimeoutInterval: 60 * 60 * 1000,
    });
    bot.once("error", onConnectionFailed);
    bot.once("login", () => console.log(`Mineflayer bot logged in with version ${bot.version}`));
    bot.once("spawn", () => console.log("Mineflayer bot spawned"));
    bot.once("end", (reason) => console.log(`Mineflayer bot ended: ${reason}`));

    // Event subscriptions
    bot.waitTicks = req.body.waitTicks;
    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];
    bot.iron_pickaxe = false;

    bot.on("kicked", (reason, loggedIn) => {
        console.log(`Mineflayer bot kicked. loggedIn=${loggedIn} reason=${reason}`);
        onDisconnect(reason);
    });

    // mounting will cause physicsTick to stop
    bot.on("mount", () => {
        bot.dismount();
    });

    bot.once("spawn", async () => {
        if (VISUAL_SERVER_PORT !== "-1") {
            try {
                console.log("Initializing Mineflayer viewer...");
                const mineflayerViewer = require('prismarine-viewer').mineflayer
                mineflayerViewer(bot, {
                    firstPerson: false,
                    viewDistance: 3,
                    port: Number(VISUAL_SERVER_PORT),
                    host: "0.0.0.0",
                });
                console.log("Mineflayer viewer initialized.");
                viewerHost = req.hostname || "127.0.0.1";
                viewerStatus = `http://127.0.0.1:${VISUAL_SERVER_PORT}`;
            } catch (error) {
                console.log(`Mineflayer viewer disabled: ${error.message}`);
                viewerStatus = `disabled: ${error.message}`;
            }
        }
        bot.removeListener("error", onConnectionFailed);
        let itemTicks = 1;
        if (req.body.reset === "hard") {
            await runCommand(bot, "/clear @s");
            const inventory = req.body.inventory ? req.body.inventory : {};
            const equipment = req.body.equipment
                ? req.body.equipment
                : [null, null, null, null, null, null];
            for (let key in inventory) {
                await runCommand(bot, `/give @s minecraft:${key} ${inventory[key]}`);
                itemTicks += 1;
            }
            const equipmentNames = [
                "armor.head",
                "armor.chest",
                "armor.legs",
                "armor.feet",
                "weapon.mainhand",
                "weapon.offhand",
            ];
            for (let i = 0; i < 6; i++) {
                if (i === 4) continue;
                if (equipment[i]) {
                    await runCommand(
                        bot,
                        `/item replace entity @s ${equipmentNames[i]} with minecraft:${equipment[i]}`
                    );
                    itemTicks += 1;
                }
            }
        }

        if (req.body.trackPlayer) {
            const playerEntity = getNearestHumanPlayerEntity(bot);
            if (playerEntity) {
                const safeTrackedPosition =
                    findSafeTrackedPosition(bot, playerEntity) ||
                    getTrackedPlayerSpawnPosition(playerEntity);
                const spawnPosition = {
                    x: Number(safeTrackedPosition.x).toFixed(2),
                    y: Math.floor(safeTrackedPosition.y),
                    z: Number(safeTrackedPosition.z).toFixed(2),
                };
                await runCommand(
                    bot,
                    `/tp @s ${spawnPosition.x} ${spawnPosition.y} ${spawnPosition.z}`
                );
                await runCommand(
                    bot,
                    `Tracking player ${playerEntity.username} from ${spawnPosition.x} ${spawnPosition.y} ${spawnPosition.z}`
                );
            } else {
                await runCommand(bot, "No human player entity found to track. Staying at current spawn.");
            }
        } else if (req.body.position) {
            await runCommand(
                bot,
                `/tp @s ${req.body.position.x} ${req.body.position.y} ${req.body.position.z}`
            );
        }

        // if iron_pickaxe is in bot's inventory
        // if (
        //     bot.inventory.items().find((item) => item.name === "iron_pickaxe")
        // ) {
        //     bot.iron_pickaxe = true;
        // }

        const {pathfinder} = require("mineflayer-pathfinder");
        const tool = require("mineflayer-tool").plugin;
        const collectBlock = require("mineflayer-collectblock").plugin;
        const pvp = require("mineflayer-pvp").plugin;
        const minecraftHawkEye = require("minecrafthawkeye");
        bot.loadPlugin(pathfinder);
        bot.loadPlugin(tool);
        bot.loadPlugin(collectBlock);
        bot.loadPlugin(pvp);
        bot.loadPlugin(minecraftHawkEye);

        // bot.collectBlock.movements.digCost = 0;
        // bot.collectBlock.movements.placeCost = 0;

        obs.inject(bot, [
            OnChat,
            OnError,
            Voxels,
            Status,
            Inventory,
            OnSave,
            Chests,
            BlockRecords,
        ]);
        skills.inject(bot);

        if (req.body.spread) {
            await runCommand(bot, `/spreadplayers ~ ~ 0 300 under 80 false @s`);
        }

        await bot.waitForTicks(bot.waitTicks * itemTicks);
        const initialObservation = bot.observe();
        initialObservation[1]["viewerStatus"] = viewerStatus;
        res.json(initialObservation);

        initCounter(bot);
        await runCommand(bot, "/gamerule keepInventory true");
        await runCommand(bot, "/gamerule doDaylightCycle false");
    });

    function onConnectionFailed(e) {
        console.log(e);
        bot = null;
        res.status(400).json({error: e});
    }

    function onDisconnect(message) {
        if (bot.viewer) {
            bot.viewer.close();
        }
        bot.end();
        console.log(message);
        bot = null;
    }
});

app.post("/step", async (req, res) => {
    // import useful package
    let response_sent = false;

    function otherError(err) {
        console.log("Uncaught Error");
        bot.emit("error", handleError(err));
        bot.waitForTicks(bot.waitTicks).then(() => {
            if (!response_sent) {
                response_sent = true;
                res.json(bot.observe());
            }
        });
    }

    process.on("uncaughtException", otherError);

    const mcData = require("minecraft-data")(bot.version);
    mcData.itemsByName["leather_cap"] = mcData.itemsByName["leather_helmet"];
    mcData.itemsByName["leather_tunic"] =
        mcData.itemsByName["leather_chestplate"];
    mcData.itemsByName["leather_pants"] =
        mcData.itemsByName["leather_leggings"];
    mcData.itemsByName["leather_boots"] = mcData.itemsByName["leather_boots"];
    mcData.itemsByName["lapis_lazuli_ore"] = mcData.itemsByName["lapis_ore"];
    mcData.blocksByName["lapis_lazuli_ore"] = mcData.blocksByName["lapis_ore"];
    const {
        Movements,
        goals: {
            Goal,
            GoalBlock,
            GoalNear,
            GoalXZ,
            GoalNearXZ,
            GoalY,
            GoalGetToBlock,
            GoalLookAtBlock,
            GoalBreakBlock,
            GoalCompositeAny,
            GoalCompositeAll,
            GoalInvert,
            GoalFollow,
            GoalPlaceBlock,
        },
        pathfinder,
        Move,
        ComputedPath,
        PartiallyComputedPath,
        XZCoordinates,
        XYZCoordinates,
        SafeBlock,
        GoalPlaceBlockOptions,
    } = require("mineflayer-pathfinder");
    const {Vec3} = require("vec3");

    // Set up pathfinder
    const movements = new Movements(bot, mcData);
    bot.pathfinder.setMovements(movements);

    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];

    function onTick() {
        bot.globalTickCounter++;
        if (bot.pathfinder.isMoving()) {
            bot.stuckTickCounter++;
            if (bot.stuckTickCounter >= 100) {
                onStuck(1.5);
                bot.stuckTickCounter = 0;
            }
        }
    }

    bot.on("physicTick", onTick);

    // initialize fail count
    let _craftItemFailCount = 0;
    let _killMobFailCount = 0;
    let _mineBlockFailCount = 0;
    let _placeItemFailCount = 0;
    let _smeltItemFailCount = 0;

    // Retrieve array form post bod
    const code = req.body.code;
    const programs = req.body.programs;
    const actionMatch = code.match(/await\s+([A-Za-z0-9_]+)\s*\(\s*bot\s*\)/);
    const actionName = actionMatch ? actionMatch[1] : "unknown";
    console.log(`ADAM_STEP_START action=${actionName} codeChars=${code.length} programsChars=${programs.length}`);
    bot.chat(`[ADAM_STEP_START] ${actionName}`);
    if (actionName.startsWith("craft")) {
        const playerEntity = getNearestHumanPlayerEntity(bot);
        const safeTrackedPosition = playerEntity ? findSafeTrackedPosition(bot, playerEntity, 8) : null;
        if (safeTrackedPosition) {
            await runCommand(
                bot,
                `/tp @s ${safeTrackedPosition.x.toFixed(2)} ${Math.floor(safeTrackedPosition.y)} ${safeTrackedPosition.z.toFixed(2)}`
            );
            bot.chat(
                `Adjusted near player to ${Math.floor(safeTrackedPosition.x)} ${Math.floor(safeTrackedPosition.y)} ${Math.floor(safeTrackedPosition.z)}`
            );
        }
    }
    bot.cumulativeObs = [];
    await bot.waitForTicks(bot.waitTicks);
    const r = await evaluateCode(code, programs);
    console.log(`ADAM_STEP_RESULT action=${actionName} result=${r === "success" ? "success" : String(r)}`);
    process.off("uncaughtException", otherError);
    if (r !== "success") {
        bot.emit("error", handleError(r));
        if (!response_sent) {
            response_sent = true;
            res.status(400).json(bot.observe());
        }
        bot.removeListener("physicTick", onTick);
        return;
    }
    await returnItems();
    // wait for last message
    await bot.waitForTicks(bot.waitTicks);
    if (!response_sent) {
        response_sent = true;
        const finalObservation = bot.observe();
        finalObservation[1]["viewerStatus"] = viewerStatus;
        res.json(finalObservation);
    }
    bot.removeListener("physicTick", onTick);

    async function evaluateCode(code, programs) {
        // Echo the code produced for players to see it. Don't echo when the bot code is already producing dialog or it will double echo
        try {
            await eval("(async () => {" + programs + "\n" + code + "})()");
            return "success";
        } catch (err) {
            return err;
        }
    }

    function onStuck(posThreshold) {
        const currentPos = bot.entity.position;
        bot.stuckPosList.push(currentPos);

        // Check if the list is full
        if (bot.stuckPosList.length === 5) {
            const oldestPos = bot.stuckPosList[0];
            const posDifference = currentPos.distanceTo(oldestPos);

            if (posDifference < posThreshold) {
                teleportBot(); // execute the function
            }

            // Remove the oldest time from the list
            bot.stuckPosList.shift();
        }
    }

    function teleportBot() {
        const blocks = bot.findBlocks({
            matching: (block) => {
                return block.type === 0;
            },
            maxDistance: 1,
            count: 27,
        });

        if (blocks) {
            // console.log(blocks.length);
            const randomIndex = Math.floor(Math.random() * blocks.length);
            const block = blocks[randomIndex];
            bot.chat(`/tp @s ${block.x} ${block.y} ${block.z}`);
        } else {
            bot.chat("/tp @s ~ ~1.25 ~");
        }
    }

    function returnItems() {
        bot.chat("/gamerule doTileDrops false");
        const crafting_table = bot.findBlock({
            matching: mcData.blocksByName.crafting_table.id,
            maxDistance: 128,
        });
        if (crafting_table) {
            bot.chat(
                `/setblock ${crafting_table.position.x} ${crafting_table.position.y} ${crafting_table.position.z} air destroy`
            );
            bot.chat("/give @s crafting_table");
        }
        const furnace = bot.findBlock({
            matching: mcData.blocksByName.furnace.id,
            maxDistance: 128,
        });
        if (furnace) {
            bot.chat(
                `/setblock ${furnace.position.x} ${furnace.position.y} ${furnace.position.z} air destroy`
            );
            bot.chat("/give @s furnace");
        }
        if (bot.inventoryUsed() >= 32) {
            // if chest is not in bot's inventory
            if (!bot.inventory.items().find((item) => item.name === "chest")) {
                bot.chat("/give @s chest");
            }
        }
        // if iron_pickaxe not in bot's inventory and bot.iron_pickaxe
        // if (
        //     bot.iron_pickaxe &&
        //     !bot.inventory.items().find((item) => item.name === "iron_pickaxe")
        // ) {
        //     bot.chat("/give @s iron_pickaxe");
        // }
        bot.chat("/gamerule doTileDrops true");
    }

    function handleError(err) {
        let stack = err.stack;
        if (!stack) {
            return err;
        }
        console.log(stack);
        const final_line = stack.split("\n")[1];
        const regex = /<anonymous>:(\d+):\d+\)/;

        const programs_length = programs.split("\n").length;
        let match_line = null;
        for (const line of stack.split("\n")) {
            const match = regex.exec(line);
            if (match) {
                const line_num = parseInt(match[1]);
                if (line_num >= programs_length) {
                    match_line = line_num - programs_length;
                    break;
                }
            }
        }
        if (!match_line) {
            return err.message;
        }
        let f_line = final_line.match(
            /\((?<file>.*):(?<line>\d+):(?<pos>\d+)\)/
        );
        if (f_line && f_line.groups && fs.existsSync(f_line.groups.file)) {
            const {file, line, pos} = f_line.groups;
            const f = fs.readFileSync(file, "utf8").split("\n");
            // let filename = file.match(/(?<=node_modules\\)(.*)/)[1];
            let source = file + `:${line}\n${f[line - 1].trim()}\n `;

            const code_source =
                "at " +
                code.split("\n")[match_line - 1].trim() +
                " in your code";
            return source + err.message + "\n" + code_source;
        } else if (
            f_line &&
            f_line.groups &&
            f_line.groups.file.includes("<anonymous>")
        ) {
            const {file, line, pos} = f_line.groups;
            let source =
                "Your code" +
                `:${match_line}\n${code.split("\n")[match_line - 1].trim()}\n `;
            let code_source = "";
            if (line < programs_length) {
                source =
                    "In your program code: " +
                    programs.split("\n")[line - 1].trim() +
                    "\n";
                code_source = `at line ${match_line}:${code
                    .split("\n")
                    [match_line - 1].trim()} in your code`;
            }
            return source + err.message + "\n" + code_source;
        }
        return err.message;
    }
});

app.post("/stop", (req, res) => {
    bot.end();
    res.json({
        message: "Bot stopped",
    });
});

app.post("/pause", (req, res) => {
    if (!bot) {
        res.status(400).json({error: "Bot not spawned"});
        return;
    }
    bot.chat("/pause");
    bot.waitForTicks(bot.waitTicks).then(() => {
        res.json({message: "Success"});
    });
});

// Server listening to PORT
const PORT = process.argv[2];
const VISUAL_SERVER_PORT = process.argv[3];
app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});
