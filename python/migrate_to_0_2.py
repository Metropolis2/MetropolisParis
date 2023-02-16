# Convert METROPOLIS 0.1.x input Agents file to METROPOLIS 0.2.x input Agents file.
import json

# Path to the old input Agents file.
OLD_FILE = "./output/old_run/agents.json"
# Path to the new input Agents file.
NEW_FILE = "./output/next_run/agents.json"

print("Reading agents")
with open(OLD_FILE) as f:
    old_agents = json.load(f)

print("Migrating agents to the new format")
new_agents = list()
for agent in old_agents:
    # Remove schedule utility parameter (not used anymore).
    if "schedule_utility" in agent:
        schedule_utility = agent.pop("schedule_utility")
    else:
        schedule_utility = None
    for mode in agent["modes"]:
        # Rename mode type.
        if mode["type"] == "Road":
            mode["type"] = "Trip"
        else:
            pass
        # Set origin / destination schedule utility.
        if not schedule_utility is None and schedule_utility["type"] == "AlphaBetaGamma":
            if schedule_utility["value"].pop("desired_arrival"):
                mode["value"]["destination_schedule_utility"] = schedule_utility
            else:
                mode["value"]["origin_schedule_utility"] = schedule_utility
        # Create a leg.
        leg = dict()
        if "travel_utility" in mode["value"]:
            travel_utility = mode["value"].pop("travel_utility")
        elif "utility_model" in mode["value"]:
            travel_utility = mode["value"].pop("utility_model")
        else:
            travel_utility = None
        if not travel_utility is None:
            if travel_utility["type"] == "None":
                travel_utility["type"] = "Polynomial"
                travel_utility["value"] = {}
            elif travel_utility["type"] == "Proportional":
                travel_utility["type"] = "Polynomial"
                travel_utility["value"] = {"b": travel_utility["value"]}
            elif travel_utility["type"] == "Quadratic":
                travel_utility["type"] = "Polynomial"
                travel_utility["value"] = {
                    "b": travel_utility["value"].get("a"),
                    "c": travel_utility["value"].get("b"),
                }
            leg["travel_utility"] = travel_utility
        leg["class"] = {
            "type": "Road",
            "value": {
                "origin": mode["value"].pop("origin"),
                "destination": mode["value"].pop("destination"),
                "vehicle": mode["value"].pop("vehicle"),
            },
        }
        mode["value"]["legs"] = [leg]
    new_agents.append(agent)


print("Writing data...")
with open(NEW_FILE, 'w') as f:
    f.write(json.dumps(new_agents))
