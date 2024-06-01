Football Analysis Projects
==========================

Overview
--------

This repository is dedicated to football data analysis, showcasing various Jupyter notebooks that detail the handling and visualization of football data from multiple perspectives. Through these projects, we explore different facets of football analytics, including creating sophisticated pass maps, evaluating player performance metrics, and more, using Python and libraries like Matplotlib and mplsoccer.


# Project Roadmap

| Field                            | Tasks                                           | Planned Date | Status  | Artefacts                                                                                      |
|----------------------------------|-------------------------------------------------|--------------|---------|------------------------------------------------------------------------------------------------|
| **Data Processing and Visualization** | Develop Pass Map Template                     | 2024-03-01   | Done ✅   | [code](https://github.com/hadjdeh/football-data-analysis/tree/main/Pass_map) [article](https://footsci.medium.com/a-detailed-guide-to-creating-advanced-pass-maps-with-python-and-matplotlib-731d6aa71a94)       |
|                                  | Scraping Static FBref Data                     | 2024-03-01   | Done ✅   | [code](https://github.com/hadjdeh/football-data-analysis/tree/main/Scraping_fbref_static_data)   |
|                                  | Develop Team Radar Template                    | 2024-03-08        | Done ✅ | [code](https://github.com/hadjdeh/football-data-analysis/tree/main/Team_radar) [article](https://footsci.medium.com/create-a-statsbomb-inspired-template-for-team-radar-comparison-using-free-data-from-fbref-1cf99c0ed0f1)                                                                                            |
|                                  | Integrate Distribution on Radar Template       | 2024-03-12        | Done ✅ | [code](https://github.com/hadjdeh/football-data-analysis/blob/main/Team_radar/notebooks/2.%20Team%20radar%20and%20distribution.ipynb)                                                                                              |
|                                  | Shots and Goals map       | 2024-03-19        | Done ✅ |    [code](https://github.com/hadjdeh/football-data-analysis/blob/main/Shots_and_goals_map/notebooks/1.Shot_and_goal_map_with_ratings.ipynb) [article](https://footsci.medium.com/plot-shots-goals-maps-with-python-mplsoccer-ranking-players-by-xg-shots-and-goals-per-90-16afa7c74b9a)                                                                                          |
|                                  | xT map by zones                        |    2024-06-01     | Planned 🔜 |                                                                                              |
|                                  | Calculating xT based on transitions matrix     |    2024-06-01     | Planned 🔜 |                                                                                              |
|                                  | Create Player Templates                        |    ❓     | To do |                                                                                              |
|                                  | Interpolate Carries on Event Data              | 2024-06-01       | Planned 🔜 |                                                                                              |
|                                  | Identify Possession Chains                     | 2024-06-01        | Planned 🔜 |                                                                                              |
|                                  | Visualize Dynamic Metric Changes (xT, xG)      | 2024-06-01        | Planned 🔜 |                                                                                              |
|                                  | TBD                                             | TBD        | TBD     |                                                                                              |
| **Data Management**               | Design data warehouse architecture             |  ❓  | To do  |                                                                                              |
|                                  | Load historical data into the warehouse        |  ❓  | To do  |                                                                                              |
|                                  | Launch regular data loading processes          |  ❓  | To do   |                                                                                              |
|                                  | Quality Assurance                               |  ❓  | To do   |                                                                                              |
|                                  | TBD                                             |    TBD          | TBD   |                                                                                              |
| **Advanced Analytics**            | Build xT transition matrix                     | 2024-03-01   | Done  ✅  | [article](https://footsci.medium.com/summary-by-expected-threat-xt-why-its-important-to-provide-transition-matrix-576cc4601395)                                                                                             |
|                                  | Build an Up-to-Date baseline VAEP Model        | ❓   | To do    |                                                                                              |
|                                  | Increasing quality of VAEP                      |   ❓ | To do   |                                                                                              |
|                                  | TBD                                             |              |         |                                                                                              |
| **Automation & Integration**      | Pilot a Twitter Bot for automated posting      | ❓   | To do   |                                                                                              |
|                                  | Pilot a Telegram Bot for automated posting     | ❓   | To do   |                                                                                              |
|                                  | TBD                                             |              |         |                                                                                              |





## 1. Pass_map 

The `Pass_map` directory contains Jupyter notebooks and datasets used for creating advanced pass maps (passing network map). Key components include:

- `1. Pass map creating v1.20240221.ipynb`: Notebook for pass map visualization.
- `data/`: Directory containing raw event data files for match Man City 1:1 Chelsea | Premier League | Season 2023-2024 | 2024-02-17
- `img/`: Directory containing resulting `pass_map.jpeg` and reference map from the Athletic teamplate the `Athletic pass map.jpeg`

For a detailed walkthrough of the pass map creation process, check out my Medium articles:

- [Article 1](https://medium.com/@footsci/passing-networks-with-expected-threat-xt-layer-7d699f75387b): Passing networks with expected threat (xT) layer. Walking through popular templates. Explaining the details.
- [Article 2](https://footsci.medium.com/a-detailed-guide-to-creating-advanced-pass-maps-with-python-and-matplotlib-731d6aa71a94): A Detailed Guide to Creating Advanced Pass Maps with Python and Matplotlib

### Results

<img src="/Pass_map/img/pass_map.jpeg" width="60%" height="auto">

## 2. Scraping_fbref_static_data

The `Scraping_fbref_static_data` directory facilitates the collection of comprehensive football statistics from FBRef, targeting the top 5 European leagues. It includes data spanning the last five seasons and up-to-date statistics for the current season (as of March 2, 2024). 

Key Components:
- `utils/`: Contains Python utility file with functions essential for data scraping and manipulation.
- `notebooks/`: Features Jupyter notebook that guides users through the scraping process (based on https://github.com/parth1902/Scrape-FBref-data/blob/master/Scrape_FBref.ipynb)
- `img/`: Provides screenshots from the FBRef website, offering insights into the tables and statistics being collected, facilitating a better understanding of the data's structure and content.
- `data/old_seasons/`: Stores historical data for the top 5 European leagues from the 2018-2019 season to the 2022-2023 season, including:
  - `top5_leagues_keeper_2018_2019__2022_2023.csv`: Goalkeeper statistics for the last five seasons.
  - `top5_leagues_outfields_2018_2019__2022_2023.csv`: Outfield player statistics.
  - `top5_leagues_team_2018_2019__2022_2023.csv`: Team-level statistics.
  - `top5_leagues_team_vs_2018_2019__2022_2023.csv`: Team versus team statistics.
- `data/current_season/{date}`/: Contains the latest season's data, structured as follows:
  - `top5_leagues_keeper_2023_2024.csv`: Current season goalkeeper statistics.
  - `top5_leagues_outfields_2023_2024.csv`: Outfield player statistics.
  - `top5_leagues_team_2023_2024.csv`: Team-level statistics.
  - `top5_leagues_team_vs_2023_2024.csv`: Team versus team statistics.
    
### Performance and Usage Advice
Data Collection Time (MacBook Air M1 8GB): Collecting the entire dataset for the last five seasons requires approximately 1.5 hours, while updating with the current season's data takes about 20 minutes (4 minute for 1 league and 1 minute if you need just Outfield data for example). This process can be expedited by leveraging multiprocessing.

Data Utilization: It is recommended to use the already available data for the past five seasons and only update with the actual data for the current season.

## 3. Team_radar

The `Team_radar` directory contains Jupyter notebooks and Python (.py) modules with utility functions used for creating a template similar to StatsBomb for generating Team Radars. The key components include:

- `notebooks/1. Team radar and statistics table.ipynb`: A notebook for team radar visualization.
- `notebooks/2. Team radar and distribution.ipynb`: A notebook for team radar visualization.
- `img/`: A directory containing the resulting radar images, statistics table images and distributions images.
- `utils/`: A directory containing modules with utility functions for creating the Radar Map

The data used for creating the template and plotting up-to-date statistics for teams are provided by the `Scraping_fbref_static_data` directory in the same repository.

For a detailed walkthrough of the process of creating team radars, check out my Medium articles:

- [Article 1](https://footsci.medium.com/create-a-statsbomb-inspired-template-for-team-radar-comparison-using-free-data-from-fbref-1cf99c0ed0f1): Create a StatsBomb-Inspired Template for Team Radar Comparison Using Free Data from FBRef


### Results

#### Radar + table with statistic

<img src="/Team_radar/img/2024-03-06/radar_image/Defending_Radar_Real%20Madrid_Barcelona_2023-2024.jpeg" width="80%" height="auto">
<img src="/Team_radar/img/2024-03-06/radar_image/Attacking_Radar_Real%20Madrid_Barcelona_2023-2024.jpeg" width="80%" height="auto">

#### Radar + distri

<img src="/Team_radar/img/2024-03-12/radar_image/Defending_Radar_dist_Napoli_2023-2024.jpeg" width="80%" height="auto">
<img src="/Team_radar/img/2024-03-12/radar_image/Defending_Radar_dist_Barcelona_2023-2024.jpeg" width="80%" height="auto">

## 3. Shots_and_goals_map

### Results

<img src="/Shots_and_goals_map/img/2024-03-18/Europe/shot_map_top9_in_Europe_by_np%20Goals.jpeg" width="80%" height="auto">
