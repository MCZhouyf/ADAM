async function craftSticks(bot) {
    const plankTypes = ["oak_planks", "birch_planks", "spruce_planks", "jungle_planks", "acacia_planks", "dark_oak_planks", "mangrove_planks"];
    for (let plankType of plankTypes) {
        let plank = bot.inventory.findInventoryItem(mcData.itemsByName[plankType].id);
        if (plank) {
            await craftItem(bot, "stick", 1);
            bot.chat(`Crafted sticks from ${plankType.replace('_', ' ')}.`);
            return;
        }
    }
}
