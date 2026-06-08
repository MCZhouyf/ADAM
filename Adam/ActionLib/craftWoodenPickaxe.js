async function craftWoodenPickaxe(bot) {
    const base = bot.entity.position.floored();
    let position = null;
    const candidateOffsets = [];
    for (let radius = 1; radius <= 4; radius++) {
        for (let dx = -radius; dx <= radius; dx++) {
            for (let dz = -radius; dz <= radius; dz++) {
                if (Math.max(Math.abs(dx), Math.abs(dz)) !== radius) {
                    continue;
                }
                candidateOffsets.push([dx, -1, dz]);
                candidateOffsets.push([dx, 0, dz]);
                candidateOffsets.push([dx, -2, dz]);
            }
        }
    }

    for (const [dx, dy, dz] of candidateOffsets) {
        const groundPos = base.offset(dx, dy, dz);
        const placePos = groundPos.offset(0, 1, 0);
        const headPos = groundPos.offset(0, 2, 0);
        const groundBlock = bot.blockAt(groundPos);
        const placeBlock = bot.blockAt(placePos);
        const headBlock = bot.blockAt(headPos);
        if (!groundBlock || groundBlock.name === "air") {
            continue;
        }
        if (!placeBlock || placeBlock.name !== "air") {
            continue;
        }
        if (headBlock && headBlock.name !== "air") {
            continue;
        }
        position = placePos;
        bot.chat(`Selected crafting table position ${position.x} ${position.y} ${position.z}`);
        break;
    }
    if (!position) {
        throw new Error(
            `No valid nearby air block found for crafting_table around ${base.x} ${base.y} ${base.z}`
        );
    }

    await placeItem(bot, "crafting_table", position);
    const placedTable = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 6,
    });
    if (!placedTable) {
        throw new Error(
            `Failed to place crafting_table near ${position.x} ${position.y} ${position.z}`
        );
    }
    await craftItem(bot, "wooden_pickaxe", 1);
    bot.save("wooden_pickaxe_crafted");
    bot.chat("Crafted a wooden pickaxe.");
}
