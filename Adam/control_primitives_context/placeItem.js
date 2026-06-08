// Place a crafting_table near the player, Vec3(1, 0, 0) is just an example, you shouldn't always use that: placeItem(bot, "crafting_table", bot.entity.position.offset(1, 0, 0));
async function placeItem(bot, name, position) {
    const item = bot.inventory.findInventoryItem(mcData.itemsByName[name].id);
    const targetBlock = bot.blockAt(position);
    if (targetBlock?.name !== "air") {
        throw new Error(`Target position for ${name} is occupied by ${targetBlock?.name}`);
    }
    // find a reference block
    const faceVectors = [
        new Vec3(0, 1, 0),
        new Vec3(0, -1, 0),
        new Vec3(1, 0, 0),
        new Vec3(-1, 0, 0),
        new Vec3(0, 0, 1),
        new Vec3(0, 0, -1),
    ];
    let referenceBlock = null;
    let faceVector = null;
    for (const vector of faceVectors) {
        const block = bot.blockAt(position.minus(vector));
        if (block?.name !== "air") {
            referenceBlock = block;
            faceVector = vector;
            break;
        }
    }
    // Only pathfind when the placement target is not already within direct reach
    if (bot.entity.position.distanceTo(position.offset(0.5, 0.5, 0.5)) > 4.5) {
        await bot.pathfinder.goto(new GoalPlaceBlock(position, bot.world, {}));
    }
    // You must equip the item right before calling placeBlock
    await bot.equip(item, "hand");
    await bot.placeBlock(referenceBlock, faceVector);
    // If Mineflayer times out waiting for blockUpdate in a local world, re-check the world state and item count before treating placement as failure.
}
