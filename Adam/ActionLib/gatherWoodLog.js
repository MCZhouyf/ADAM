async function gatherWoodLog(bot) {
  const { goals } = require("mineflayer-pathfinder");
  const { GoalNear } = goals;
  const mcData = require("minecraft-data")(bot.version);
  const debugChatEnabled = process.env.ADAM_DEBUG_CHAT === "1";
  const startedAt = Date.now();
  const actionTimeoutMs = 150000;

  bot.chat("Gathering wood logs started near current position");

  const logNames = [
    "oak_log",
    "birch_log",
    "spruce_log",
    "jungle_log",
    "acacia_log",
    "dark_oak_log",
    "mangrove_log",
  ];

  function summarizeInventory() {
    return bot.inventory.items().map(item => ({
      name: item.name,
      count: item.count,
      slot: item.slot,
    }));
  }

  function inventoryToDict(items) {
    const result = {};
    for (const item of items) {
      if (!item || !item.name || !item.count) {
        continue;
      }
      result[item.name] = (result[item.name] || 0) + item.count;
    }
    return result;
  }

  function diffInventory(beforeItems, afterItems) {
    const before = inventoryToDict(beforeItems);
    const after = inventoryToDict(afterItems);
    const added = {};
    const removed = {};
    const keys = new Set([...Object.keys(before), ...Object.keys(after)]);

    for (const key of keys) {
      const beforeCount = before[key] || 0;
      const afterCount = after[key] || 0;
      if (afterCount > beforeCount) {
        added[key] = afterCount - beforeCount;
      } else if (beforeCount > afterCount) {
        removed[key] = beforeCount - afterCount;
      }
    }

    return { added, removed };
  }

  function summarizeBlock(block) {
    if (!block) {
      return null;
    }
    return {
      name: block.name,
      type: block.type,
      stateId: block.stateId,
      metadata: block.metadata,
      position: {
        x: block.position.x,
        y: block.position.y,
        z: block.position.z,
      },
      drops:
        block.drops && Array.isArray(block.drops)
          ? block.drops.map(drop => ({
              id: drop.id,
              metadata: drop.metadata,
              count: drop.count,
              name: drop.name || null,
            }))
          : [],
      diggable: block.diggable,
      boundingBox: block.boundingBox,
      material: block.material || null,
      transparent: block.transparent,
      harvestTools: block.harvestTools || null,
    };
  }

  function debugMessage(message) {
    console.log(message);
    if (debugChatEnabled) {
      try {
        bot.chat(message);
      } catch (error) {
      }
    }
  }

  function isTimedOut() {
    return Date.now() - startedAt > actionTimeoutMs;
  }

  function withTimeout(promise, timeoutMs, label) {
    return Promise.race([
      promise,
      new Promise((_, reject) => {
        setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
      }),
    ]);
  }

  async function runCommand(command, extraTicks = 2) {
    debugMessage(`[ADAM_DEBUG][gatherWoodLog] command=${command}`);
    bot.chat(command);
    await bot.waitForTicks(bot.waitTicks * extraTicks);
  }

  function summarizeNearbyDrops(radius = 8) {
    return Object.values(bot.entities)
      .filter(entity => {
        return (
          entity &&
          entity.name === "item" &&
          entity.position &&
          entity.position.distanceTo(bot.entity.position) <= radius
        );
      })
      .map(entity => ({
        item: (() => {
          const itemMeta = entity.metadata && entity.metadata[8];
          if (!itemMeta || !itemMeta.present) {
            return null;
          }
          const itemInfo = mcData.items[itemMeta.itemId] || null;
          return {
            id: itemMeta.itemId,
            name: itemInfo ? itemInfo.name : null,
            count: itemMeta.itemCount,
          };
        })(),
        id: entity.id,
        position: {
          x: Number(entity.position.x.toFixed(2)),
          y: Number(entity.position.y.toFixed(2)),
          z: Number(entity.position.z.toFixed(2)),
        },
        velocity: entity.velocity
          ? {
              x: Number(entity.velocity.x.toFixed(2)),
              y: Number(entity.velocity.y.toFixed(2)),
              z: Number(entity.velocity.z.toFixed(2)),
            }
          : null,
        metadata: entity.metadata || null,
      }));
  }

  async function waitForInventoryChange(previousTotalCount, maxChecks = 4, waitTicks = 10) {
    for (let index = 0; index < maxChecks; index += 1) {
      if (isTimedOut()) {
        debugMessage("[ADAM_DEBUG][gatherWoodLog] wait_inventory_timeout");
        return false;
      }
      await bot.waitForTicks(waitTicks);
      const inventoryNow = summarizeInventory();
      const totalCount = inventoryNow.reduce((sum, item) => sum + item.count, 0);
      const nearbyDrops = summarizeNearbyDrops();
      debugMessage(
        `[ADAM_DEBUG][gatherWoodLog] poll=${index + 1}/${maxChecks} inventory=${JSON.stringify(inventoryNow)} nearbyDrops=${JSON.stringify(nearbyDrops)}`
      );
      if (totalCount > previousTotalCount) {
        return true;
      }
      if (nearbyDrops.length) {
        const closestDrop = Object.values(bot.entities)
          .filter(entity => entity && entity.name === "item" && entity.position)
          .sort(
            (a, b) =>
              a.position.distanceTo(bot.entity.position) -
              b.position.distanceTo(bot.entity.position)
          )[0];
        if (closestDrop) {
          try {
            await withTimeout(
              bot.pathfinder.goto(
              new GoalNear(
                Math.floor(closestDrop.position.x),
                Math.floor(closestDrop.position.y),
                Math.floor(closestDrop.position.z),
                1
              )
              ),
              8000,
              "pickup goto"
            );
          } catch (error) {
            bot.chat(`Pickup path failed near drop ${closestDrop.id}`);
          }
        }
      }
    }
    return false;
  }

  function getTargetItemNamesForBlock(blockName) {
    if (!blockName || !blockName.endsWith("_log")) {
      return [];
    }
    return [blockName];
  }

  function hasExpectedItems(addedItems, expectedItemNames) {
    return expectedItemNames.some(itemName => (addedItems[itemName] || 0) > 0);
  }

  function getNearbyDropEntities(radius = 8) {
    return Object.values(bot.entities)
      .filter(entity => {
        return (
          entity &&
          entity.name === "item" &&
          entity.position &&
          entity.position.distanceTo(bot.entity.position) <= radius
        );
      });
  }

  async function confirmPickupOrTimeout({
    inventoryBefore,
    targetBlockName,
    maxChecks = 4,
    waitTicks = 10,
  }) {
    const beforeTotal = inventoryBefore.reduce((sum, item) => sum + item.count, 0);
    const expectedItemNames = getTargetItemNamesForBlock(targetBlockName);
    const initialDropIds = new Set(getNearbyDropEntities().map(entity => entity.id));
    let lastInventory = inventoryBefore;
    let lastDelta = { added: {}, removed: {} };
    let dropEntityConsumed = initialDropIds.size === 0;
    let expectedItemPicked = false;

    for (let index = 0; index < maxChecks; index += 1) {
      if (isTimedOut()) {
        debugMessage("[ADAM_DEBUG][gatherWoodLog] confirm_pickup_timeout");
        break;
      }
      await bot.waitForTicks(waitTicks);
      const currentInventory = summarizeInventory();
      lastInventory = currentInventory;
      lastDelta = diffInventory(inventoryBefore, currentInventory);
      const currentDropEntities = getNearbyDropEntities();
      const currentDropIds = new Set(currentDropEntities.map(entity => entity.id));
      dropEntityConsumed =
        initialDropIds.size > 0 &&
        [...initialDropIds].every(id => !currentDropIds.has(id));
      expectedItemPicked = hasExpectedItems(lastDelta.added, expectedItemNames);

      debugMessage(
        `[ADAM_DEBUG][gatherWoodLog] confirm_pickup=${index + 1}/${maxChecks} expected=${JSON.stringify(expectedItemNames)} added=${JSON.stringify(lastDelta.added)} dropEntityConsumed=${dropEntityConsumed} currentDrops=${JSON.stringify(
          currentDropEntities.map(entity => ({
            id: entity.id,
            x: Number(entity.position.x.toFixed(2)),
            y: Number(entity.position.y.toFixed(2)),
            z: Number(entity.position.z.toFixed(2)),
          }))
        )}`
      );

      const totalCount = currentInventory.reduce((sum, item) => sum + item.count, 0);
      if (!expectedItemPicked && totalCount > beforeTotal && currentDropEntities.length) {
        const closestDrop = currentDropEntities
          .sort(
            (a, b) =>
              a.position.distanceTo(bot.entity.position) -
              b.position.distanceTo(bot.entity.position)
          )[0];
        try {
          await withTimeout(
            bot.pathfinder.goto(
            new GoalNear(
              Math.floor(closestDrop.position.x),
              Math.floor(closestDrop.position.y),
              Math.floor(closestDrop.position.z),
              1
            )
            ),
            8000,
            "confirm pickup goto"
          );
        } catch (error) {
          debugMessage(
            `[ADAM_DEBUG][gatherWoodLog] pickup_retry_failed drop=${closestDrop.id} error=${error.message}`
          );
        }
      }

      if (expectedItemPicked || dropEntityConsumed) {
        return {
          success: expectedItemPicked,
          dropEntityConsumed,
          inventoryAfter: currentInventory,
          inventoryDelta: lastDelta,
        };
      }
    }

    return {
      success: false,
      dropEntityConsumed,
      inventoryAfter: lastInventory,
      inventoryDelta: lastDelta,
    };
  }

  function getNearbyLogBlocks(radius, count = 24) {
    return bot.findBlocks({
      matching: block => logNames.includes(block.name),
      maxDistance: radius,
      count,
    })
      .map(position => bot.blockAt(position))
      .filter(block => block && logNames.includes(block.name))
      .sort(
        (a, b) =>
          a.position.distanceTo(bot.entity.position) -
          b.position.distanceTo(bot.entity.position)
      );
  }

  let targets = getNearbyLogBlocks(20, 12);

  if (!targets.length) {
    bot.chat("No nearby tree in 20 blocks. Starting a visible search.");
    const searchOffsets = [
      { x: 12, z: 0 },
      { x: 0, z: 12 },
      { x: -12, z: 0 },
      { x: 0, z: -12 },
      { x: 12, z: 12 },
      { x: -12, z: 12 },
      { x: -12, z: -12 },
      { x: 12, z: -12 },
    ];

    for (const offset of searchOffsets) {
      const goalX = Math.floor(bot.entity.position.x + offset.x);
      const goalY = Math.floor(bot.entity.position.y);
      const goalZ = Math.floor(bot.entity.position.z + offset.z);
      bot.chat(`Searching toward ${goalX} ${goalY} ${goalZ}`);
      try {
        await bot.pathfinder.goto(new GoalNear(goalX, goalY, goalZ, 2));
      } catch (error) {
        bot.chat(`Search step failed near ${goalX} ${goalZ}`);
      }

      targets = getNearbyLogBlocks(24, 12);
      if (targets.length) {
        break;
      }
    }
  }
  if (!targets.length) {
    bot.chat("No reachable wood log found after visible search. Move the player/bot near a tree or set ADAM_BOT_POSITION near trees.");
    return;
  }

  const mineTargets = targets.slice(0, 1);
  bot.chat(`Mining up to ${mineTargets.length} nearby wood logs.`);

  let minedTargetCount = 0;
  let inventoryProgressCount = 0;
  let expectedPickupCount = 0;
  let sideDropOnlyCount = 0;
  let noProgressCount = 0;
  for (const target of mineTargets) {
    if (isTimedOut()) {
      debugMessage("[ADAM_DEBUG][gatherWoodLog] action_timeout_before_next_target");
      break;
    }
    const liveTarget = bot.blockAt(target.position);
    if (!liveTarget || !logNames.includes(liveTarget.name)) {
      continue;
    }

    try {
      const inventoryBefore = summarizeInventory();
      const inventoryBeforeTotal = inventoryBefore.reduce((sum, item) => sum + item.count, 0);
      debugMessage(
        `[ADAM_DEBUG][gatherWoodLog] before_target=${liveTarget.position.x},${liveTarget.position.y},${liveTarget.position.z} inventory=${JSON.stringify(inventoryBefore)}`
      );
      debugMessage(
        `[ADAM_DEBUG][gatherWoodLog] target_block=${JSON.stringify(summarizeBlock(liveTarget))}`
      );
      const below = bot.blockAt(liveTarget.position.offset(0, -1, 0));
      const above = bot.blockAt(liveTarget.position.offset(0, 1, 0));
      debugMessage(
        `[ADAM_DEBUG][gatherWoodLog] target_neighbors below=${below ? below.name : "null"} above=${above ? above.name : "null"}`
      );
      debugMessage(
        `[ADAM_DEBUG][gatherWoodLog] gameMode=${bot.game ? bot.game.gameMode : "unknown"}`
      );
      await runCommand("/gamemode survival @s");
      await runCommand("/gamerule doTileDrops true");
      await runCommand("/gamerule doTileDrops");

      await withTimeout(
        bot.pathfinder.goto(
        new GoalNear(
          liveTarget.position.x,
          liveTarget.position.y,
          liveTarget.position.z,
          1
        )
        ),
        10000,
        "target goto"
      );
      await bot.lookAt(liveTarget.position.offset(0.5, 0.5, 0.5), true);
      await withTimeout(bot.dig(liveTarget), 8000, "dig");
      minedTargetCount += 1;
      bot.chat(
        `Mined target block ${liveTarget.name} at ${liveTarget.position.x} ${liveTarget.position.y} ${liveTarget.position.z}`
      );
      await bot.waitForTicks(bot.waitTicks * 2);

      const nearbyDropsAfterDig = summarizeNearbyDrops();
      debugMessage(
        `[ADAM_DEBUG][gatherWoodLog] drops_after_dig=${JSON.stringify(nearbyDropsAfterDig)}`
      );

      if (nearbyDropsAfterDig.length) {
        for (const drop of nearbyDropsAfterDig) {
          try {
            await withTimeout(
              bot.pathfinder.goto(
              new GoalNear(
                Math.floor(drop.position.x),
                Math.floor(drop.position.y),
                Math.floor(drop.position.z),
                1
              )
              ),
              8000,
              "drop goto"
            );
            await bot.waitForTicks(bot.waitTicks);
          } catch (error) {
            bot.chat(`Skipping unreachable drop ${drop.id}`);
          }
        }
      }

      const gotInventoryProgress = await waitForInventoryChange(inventoryBeforeTotal, 12, 10);
      const pickupResult = await confirmPickupOrTimeout({
        inventoryBefore,
        targetBlockName: liveTarget.name,
        maxChecks: 12,
        waitTicks: 10,
      });
      const inventoryAfter = pickupResult.inventoryAfter;
      const inventoryDelta = pickupResult.inventoryDelta;
      debugMessage(
        `[ADAM_DEBUG][gatherWoodLog] inventory_delta target=${liveTarget.position.x},${liveTarget.position.y},${liveTarget.position.z} added=${JSON.stringify(inventoryDelta.added)} removed=${JSON.stringify(inventoryDelta.removed)}`
      );
      if (pickupResult.success) {
        inventoryProgressCount += 1;
        expectedPickupCount += 1;
        bot.chat(
          `Picked expected item after mining ${liveTarget.name}: added=${JSON.stringify(inventoryDelta.added)} removed=${JSON.stringify(inventoryDelta.removed)}`
        );
      } else if (gotInventoryProgress || Object.keys(inventoryDelta.added).length > 0) {
        sideDropOnlyCount += 1;
        bot.chat(
          `Side-drop only after mining ${liveTarget.name}: added=${JSON.stringify(inventoryDelta.added)} removed=${JSON.stringify(inventoryDelta.removed)}`
        );
      } else {
        noProgressCount += 1;
        debugMessage(
          `[ADAM_DEBUG][gatherWoodLog] no_inventory_progress target=${liveTarget.position.x},${liveTarget.position.y},${liveTarget.position.z}`
        );
        bot.chat(
          `No inventory progress after mining ${liveTarget.name} at ${liveTarget.position.x} ${liveTarget.position.y} ${liveTarget.position.z}`
        );
      }
      break;
    } catch (error) {
      bot.chat(`Skipping unreachable log at ${liveTarget.position.x} ${liveTarget.position.y} ${liveTarget.position.z}: ${error.message}`);
      debugMessage(
        `[ADAM_DEBUG][gatherWoodLog] target_error=${liveTarget.position.x},${liveTarget.position.y},${liveTarget.position.z} error=${error.stack || error.message}`
      );
    }
  }

  if (!minedTargetCount) {
    bot.chat("Found nearby logs, but none were reachable for mining.");
    return;
  }

  try {
    const rawInventory = bot.inventory.items().map(item => ({
      name: item.name,
      count: item.count,
      slot: item.slot,
    }));
    bot.chat(`[ADAM_DEBUG] Inventory after gatherWoodLog: ${JSON.stringify(rawInventory)}`);
    console.log(`[ADAM_DEBUG][gatherWoodLog] inventory_after=${JSON.stringify(rawInventory)}`);
  } catch (error) {
    bot.chat(`[ADAM_DEBUG] Failed to read inventory after gatherWoodLog: ${error.message}`);
    console.log(`[ADAM_DEBUG][gatherWoodLog] inventory_after_error=${error.message}`);
  }

  let saveMarker = "wood_log_gathered:none";
  if (expectedPickupCount > 0) {
    saveMarker = "wood_log_gathered:expected_item_picked";
  } else if (sideDropOnlyCount > 0) {
    saveMarker = "wood_log_gathered:side_drop_only";
  } else if (noProgressCount > 0) {
    saveMarker = "wood_log_gathered:no_progress";
  }

  bot.save(saveMarker);
  bot.chat(
    `Mined ${minedTargetCount} target log blocks. Inventory progress observed on ${inventoryProgressCount}. ` +
    `Expected pickups=${expectedPickupCount}, sideDrops=${sideDropOnlyCount}, noProgress=${noProgressCount}.`
  );
}
