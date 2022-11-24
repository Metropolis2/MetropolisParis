# Application of Metropolis to Paris's urban area

This repository contains a set of script files that can be used to build and run a simulaton of Metropolis for the Île-de-France region.

## How to run

The simulation input is built by running sequentially Python scripts.
Each Python script can be configured through global variables at the beginning of the files (the variables in capital letters).

### Step 1: Pre-process the road network

- Script: `python/get_here_network.py`
- Input: HERE Streets Shapefile
- Output: FlatGeobuf file with node ids and geometries and FlatGeobuf file with edge ids, geometries and characteristics

The nodes and edges of the road network are read from a HERE Shapefile.
The resulting graph has 177152 nodes and 308027 edges.

Details:

- The edges with `FUNC_CLASS <= 4` are selected.
- The number of lanes for edges whose given number of lanes is 0 is set to 1.
- Only the largest weakly connected component of the resulting graph is kept (i.e., isolated edges are removed).

### Step 2: Create connectors

- Script: `python/create_connectors.py`
- Input: The two input files from Step 1 and the IGN Shapefile of IRIS zones
- Output:
  - CSV file that maps IRIS zone ids to road-network node ids
  - FlatGeobuf with node ids and geometries (including the nodes representing zones)
  - FlatGeobuf with edge ids, geometries and characteristics (including the in/out/direct connectors)

The trip origins and destinations are located at the IRIS level.
To load and unload the agents in the road network, we use virtual nodes to represent the starting and ending point of the trips and we use virtual edges (or connectors) to connect this virtual nodes to the actual road network.

There are two virtual nodes for each zone: _in virtual nodes_ used as the starting point of the trips (with only outgoing connectors) and _out virtual nodes_ used as the ending point of the trips (with only incoming connectors).
This is a trick to prevent agents in Metropolis from going through a virtual node.

There are two types of connectors:

- In connectors that connect from a node of the road network to a (in) virtual node.
- Out connectors that connect from a (out) virtual node to a node of the road network.

Details:

- For each zone which contains at least 5 nodes, the location of the virtual nodes representing the zone is set to the [medoid](https://en.wikipedia.org/wiki/Medoid) of all the nodes in the zone.
- For each zone which contains less than 5 nodes, the location of the virtual nodes representing the zone is set to the centroid of the zone polygon.
- For each zone, at most 4 in connectors and 4 out connectors can be created, one for each bearing (North, West, South, Est).
- The distance of a node to a zone is computed as the distance between this node and the polygon of the zone (it is zero if the node is in the zone polygon).
- The nodes to connect to are selected among the nodes whose distance to the zone is not larger than the distance to the zone of the node that is the 20th closest to the zone.
- For each zone, in each bearing, the node to connect _from_ for the _in connector_ is the node that maximizes the distance to the virtual node among the nodes that minimizes the distance to the zone, among the nodes selected for this zone which have at least one _incoming_ edge.
- For each zone, in each bearing, the node to connect _to_ for the _out connector_ is the node that maximizes the distance to the virtual node among the nodes that minimizes the distance to the zone, among the nodes selected for this zone which have at least one _outgoing_ edge.
- When the connected node is in the zone and there are at least 5 nodes in the zone, the distance between the virtual node and the connected node is set to the average euclidian distance between the connected node and the other nodes in the zone.
- In any other case, the distance between the virtual node and the connected node is set to the euclidian distance between the connected node and the polygon of the zone, plus the square root of the area of the zone divided by π.
- Connectors are assumed to have 1 lane and a speed limit of 30 km/h.

### Step 3: Generate the population

- Script: `python/generate_population.py`
- Input:
  - Output of the [synthetic population generation](https://github.com/eqasim-org/ile-de-france)
  - IGN Shapefile of IRIS zones
  - CSV file that maps IRIS zone ids to road-network node ids (from Step 2)
- Output: CSV file with a list of trip, including the origin / destination IRIS zone and node id and the characteristics of the agent

The trips are generated using a synthetic population for Île-de-France (built with various datasets such as INSEE census, transport surveys and infrastructure databases).

### Step 4: Filter the population

- Script: `python/filter_population.py`
- Input: CSV file with a list of trips (from Step 3)
- Output: CSV file with the filtered list of trips

The trips generated in Step 3 are filtered to keep only the trips that should be simulated.
Trips can be filtered according to departure time, arrival time and mode.
Optionnally, the intra-zonal trips (trips whose origin and destination zone is the same) can be removed and only the longest trip of each individual can be kept.

Details:

- We filter trips by car that occurs within the time window 3AM--10AM.
- Intra-zonal trips are removed because they are not used in Metropolis (they represent 7.73 % of total travel time, mostly because of round trips).
- For each person, only the trip with the largest travel time is kept (removing 9.36 % of remaining total travel time).
  When Metropolis will be able to run with intermediary stops, these trips could be kept.

### Step 5: Generate Metropolis input

- Script: `python/write_metropolis_input.py`
- Input:
  - CSV file with the list of trips to generate (from Step 4)
  - FlatGeobuf file with the edges of the road network (from Step 2)
- Output:
  - JSON file with the characteristics of the network, ready to be read by the Metropolis simulator
  - JSON file with the characteristics of the agents, ready to be read by the Metropolis simulator

To run the Metropolis simulator, three JSON files are required with the agents, the network and the parameters.
(In the future, the Metropolis simulator will be able to run from an interface that does not require the user to write these raw files.)

The behavior of the script can be changed through many global variables:

- Length and passenger car equivalent of the agents' vehicle (they all have the same vehicle type).
- Simulated period.
- Capacity of the entry and exit bottlenecks of the edge (and a boolean to enable/disable them).
- Parameters of the alpha-beta-gamma model.
- Whether to use exogenous departure time (from the synthetic population input) or endogenous departure time (with a given μ).

Details:

- The speed-density function of the edges is set to `FreeFlow` (for now).
- The desired arrival time of the agents for their trip is set to the arrival time reported from the synthetic population.
- Mode choice is disabled: all agents travel by car.

### Extra script: Read Metropolis output

- Script: `python/read_metropolis_output.py`
- Input:
  - Compressed JSON file with the agent-specific results (from the simulator Metropolis)
  - FlatGeobuf file with the edges of the road network (from Step 2)

This script provides functions to analyse the output of the simulator.

Currently, the only function provided is:

- `get_agent_map`: Function to draw a map with the path taken by an agent during the last iteration. The color of the edges represent the congestion.
