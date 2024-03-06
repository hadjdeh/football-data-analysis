Football Analysis Projects
==========================

Overview
--------

This repository is dedicated to football data analysis, showcasing various Jupyter notebooks that detail the handling and visualization of football data from multiple perspectives. Through these projects, we explore different facets of football analytics, including creating sophisticated pass maps, evaluating player performance metrics, and more, using Python and libraries like Matplotlib and mplsoccer.


# Project Roadmap


| Goals | Description | Plan             | Status |
|-------|-------------|------------------|--------|
| Examples of Data Processing and Visualization |1)Develop Team Radar Template<br> 2)Integrate Distribution on Radar Template<br>3)Create Player Templates<br>4)Interpolation of Carries on Event Data<br>5)Identification of Possession Chains<br>6)TBD|
| Data Management	 | 1)Design data warehouse architecture for integrating football data sources<br>2)Load historical data into the warehouse<br>Launch regular data loading processes<br>TBD  |
|  Advanced Analytics | 1) Build xT transition matrix<br>2)Build an Up-to-Date VAEP Model<br>3)Increasing quality of VAEP |
| Innovation and Dissemination | - Pilot a Twitter Bot for automated posting |
|  Community Engagement | - Foster a community of contributors |
| | Quality Assurance | - Implement robust testing and validation processes |
| | Documentation and Tutorials | - Develop comprehensive guides and tutorials |


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

