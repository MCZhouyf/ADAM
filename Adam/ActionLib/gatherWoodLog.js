async function gatherWoodLog(bot) {
  const { goals } = require("mineflayer-pathfinder");
  const { GoalNear } = goals;

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

  const mineTargets = targets.slice(0, 4);
  bot.chat(`Mining up to ${mineTargets.length} nearby wood logs.`);

  let collectedCount = 0;
  for (const target of mineTargets) {
    const liveTarget = bot.blockAt(target.position);
    if (!liveTarget || !logNames.includes(liveTarget.name)) {
      continue;
    }

    try {
      await bot.collectBlock.collect(liveTarget, {
        ignoreNoPath: true,
        count: 1,
      });
      collectedCount += 1;
      bot.chat(`Collected ${liveTarget.name} at ${liveTarget.position.x} ${liveTarget.position.y} ${liveTarget.position.z}`);
    } catch (error) {
      bot.chat(`Skipping unreachable log at ${liveTarget.position.x} ${liveTarget.position.y} ${liveTarget.position.z}`);
    }
  }

  if (!collectedCount) {
    bot.chat("Found nearby logs, but none were reachable for mining.");
    return;
  }

  bot.save("wood_log_gathered");
  bot.chat(`Gathered ${collectedCount} nearby wood logs.`);
}
