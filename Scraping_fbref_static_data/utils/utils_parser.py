import requests
from bs4 import BeautifulSoup
import re
from tqdm.notebook import tqdm
import time
import pandas as pd
from functools import reduce


def get_tables_from_competitions(url):
    """
    Fetches competition tables from a given URL.
    
    Parameters:
    - url: The URL from which to fetch the tables.
    
    Returns:
    - A tuple containing player_table, team_table, and team_table_vs.
    """
    res = requests.get(url)  # Send a GET request to the URL
    time.sleep(10)  # Pause for a second to avoid overwhelming the server
    comm = re.compile("<!--|--!>")  # Regex to find comments
    soup = BeautifulSoup(comm.sub("", res.text), 'lxml')  # Parse the HTML, removing comments
    all_tables = soup.findAll("tbody")  # Find all table bodies
    team_table = all_tables[0]  # First table is team data
    team_table_vs = all_tables[1]  # Second table is team vs team data
    player_table = all_tables[2]  # Third table is player data
    
    return player_table, team_table, team_table_vs


def get_column_names_from_competitions(top, end, category, type_table):
    """
    Constructs the URL and fetches column names for the specified table type.
    
    Parameters:
    - top, end, category: Components to construct the URL.
    - type_table: The type of table ('player', 'team', 'team_vs').
    
    Returns:
    - A list of column names from the specified table.
    """
    url = (top + category + end)  # Construct the URL
    player_table, team_table, team_table_vs = get_tables_from_competitions(url)  # Fetch tables
    
    # Select the table based on type_table parameter
    if type_table == 'player':
        table = player_table
    elif type_table == 'team':
        table = team_table
    elif type_table == 'team_vs':
        table = team_table_vs
        
    lst = []  # List to hold column names
    rows = table.find_all('tr')  # Find all rows in the table
    
    # Attempt to extract column names from the first row
    for row in rows[0].findAll():
        try:
            column_name = row['data-stat']
            lst.append(row['data-stat'])
        except Exception as e:
            pass  # Ignore any errors and continue
    return lst


def get_frame(features, player_table):
    """
    Extracts data from the player table and constructs a DataFrame.
    
    Parameters:
    - features: A list of features (column names) to extract.
    - player_table: The BeautifulSoup object containing player data.
    
    Returns:
    - A pandas DataFrame constructed from the player table data.
    """
    pre_df_player = dict()  # Dictionary to hold data before creating DataFrame
    features_wanted_player = features  # Features to include in the DataFrame
    rows_player = player_table.find_all('tr')  # Find all rows in the player table
    
    # Process each row in the table
    for row in rows_player:
        if(row.find('th', {"scope":"row"}) != None):
            for f in features_wanted_player:
                try:
                    cell = row.find("td", {"data-stat": f})
                    a = cell.text.strip().encode()
                    text = a.decode("utf-8")
                    # Handle empty cells
                    if(text == ''):
                        text = '0'
                    # Convert numeric values from strings
                    if(f not in ['player', 'nationality', 'position', 'squad', 'age', 'birth_year', 'team']):
                        text = float(text.replace(',', ''))
                    # Add data to the dictionary
                    if f in pre_df_player:
                        pre_df_player[f].append(text)
                    else:
                        pre_df_player[f] = [text]
                except Exception as e:
                    pass  # Ignore errors and continue
    
    df_player = pd.DataFrame.from_dict(pre_df_player)  # Convert dictionary to DataFrame
    return df_player


def get_frame_team(features, team_table):
    """
    Extracts data from the team table and constructs a DataFrame.
    
    Parameters:
    - features: A list of features (column names) to extract.
    - team_table: The BeautifulSoup object containing team data.
    
    Returns:
    - A pandas DataFrame constructed from the team table data.
    """
    pre_df_squad = dict()  # Dictionary to hold data before creating DataFrame
    features_wanted_squad = features  # Features to include in the DataFrame
    rows_squad = team_table.find_all('tr')  # Find all rows in the team table
    
    # Process each row in the table
    for row in rows_squad:
        if(row.find('th', {"scope":"row"}) != None):
            name = row.find('th', {"data-stat": "team"}).text.strip().encode().decode("utf-8")
            # Handle the squad name separately
            if 'squad' in pre_df_squad:
                pre_df_squad['squad'].append(name)
            else:
                pre_df_squad['squad'] = [name]
            # Process each feature wanted
            for f in features_wanted_squad:
                try:
                    cell = row.find("td", {"data-stat": f})
                    a = cell.text.strip().encode()
                    text = a.decode("utf-8")
                    # Handle empty cells
                    if(text == ''):
                        text = '0'
                    # Convert numeric values from strings
                    if(f not in ['player', 'nationality', 'position', 'squad', 'age', 'birth_year']):
                        text = float(text.replace(',', ''))
                    # Add data to the dictionary
                    if f in pre_df_squad:
                        pre_df_squad[f].append(text)
                    else:
                        pre_df_squad[f] = [text]
                except Exception as e:
                    pass  # Ignore errors and continue
    
    df_squad = pd.DataFrame.from_dict(pre_df_squad)  # Convert dictionary to DataFrame
    return df_squad

def frame_for_category(category, top, end, features):
    """
    Fetches and constructs a DataFrame for a specific category of player data.
    
    Parameters:
    - category: The category of data to fetch (e.g., 'stats', 'shooting').
    - top: The base URL.
    - end: The specific endpoint or URL suffix for the category.
    - features: List of features (column names) to be included in the DataFrame.
    
    Returns:
    - DataFrame containing player data for the specified category.
    """
    url = (top + category + end)  # Construct the URL
    # Fetch tables from the URL
    player_table, _, _ = get_tables_from_competitions(url)
    # Construct and return the DataFrame for player data
    df_player = get_frame(features, player_table)
    return df_player

def frame_for_category_team(category, top, end, features):
    """
    Similar to frame_for_category but specifically for team data.
    
    Parameters are identical to frame_for_category, but this function focuses on team table.
    
    Returns:
    - DataFrame containing team data for the specified category.
    """
    url = (top + category + end)
    _, team_table, _ = get_tables_from_competitions(url)
    df_team = get_frame_team(features, team_table)
    return df_team

def frame_for_category_team_vs(category, top, end, features):
    """
    Similar to frame_for_category but for team versus team data.
    
    Parameters are identical to frame_for_category, focusing on team vs team table.
    
    Returns:
    - DataFrame containing team vs team data for the specified category.
    """
    url = (top + category + end)
    _, _, team_table_vs = get_tables_from_competitions(url)
    df_team = get_frame_team(features, team_table_vs)
    return df_team

def get_outfield_data(top, end, dict_res, type_table='player'):
    """
    Fetches and merges outfield player data across multiple categories into a single DataFrame.
    
    Parameters:
    - top: The base URL.
    - end: The specific endpoint or URL suffix.
    - dict_res: A dictionary specifying features to include for each category.
    - type_table: Specifies the type of data ('player' by default).
    
    Returns:
    - Merged DataFrame containing outfield player data across specified categories.
    """
    # Fetch DataFrames for each category and store in a list
    data_frames = [frame_for_category(category, top, end, dict_res[type_table][category]) for category in dict_res[type_table]]
    # Merge all DataFrames on 'player' and 'team', resolving suffixes and duplicates
    df = reduce(lambda left, right: pd.merge(left, right, on=['player', 'team'], how='outer', suffixes=('', 'y_right')), data_frames)
    # Remove columns with 'y_right' in their names (resulting from merge conflicts)
    df = df.loc[:, ~df.columns.duplicated(keep='last')].drop(columns=[col for col in df if 'y_right' in col])
    print(f'get_outfield_data for {end} is loaded')
    return df

def get_keeper_data(top, end, dict_res, type_table='player'):
    """
    Fetches and merges goalkeeper data across specified categories into a single DataFrame.
    
    This function is similar to get_outfield_data but tailored for goalkeeper data.
    
    Parameters and return value are identical to get_outfield_data.
    """
    # Fetch and merge DataFrames for goalkeeper categories
    data_frames = [frame_for_category(category, top, end, dict_res[type_table][category]) for category in ('keepers', 'keepersadv')]
    df = reduce(lambda left, right: pd.merge(left, right, on=['player', 'team'], how='outer', suffixes=('', 'x_right')), data_frames)
    # Clean up merged DataFrame
    df = df.loc[:, ~df.columns.duplicated(keep='last')].drop(columns=[col for col in df if 'x_right' in col])
    print(f'get_keeper_data for {end} is loaded')
    return df

def get_team_data(top, end, dict_res, type_table='team'):
    """
    Fetches and combines team data across various statistical categories into a single DataFrame.
    
    Parameters:
    - top: Base URL part before category specification.
    - end: URL part after the category, typically parameters or filters.
    - dict_res: A dictionary specifying the features to retrieve for each category.
    - type_table: Specifies the type of table data to fetch, defaulting to 'team'.
    
    Returns:
    - A pandas DataFrame containing merged data across specified categories for teams.
    """
    
    # Fetch data for each category specified in dict_res[type_table]
    # and store each resulting DataFrame in a list.
    data_frames = [
        frame_for_category_team(category, top, end, dict_res[type_table][category])
        for category in ['stats', 'keepers', 'keepersadv', 'shooting', 'passing',
                         'passing_types', 'gca', 'defense', 'possession', 'misc', 'playingtime']
    ]
    
    # Merge all DataFrames on the 'squad' column, resolving any suffix conflicts by removing '_right'.
    df = reduce(lambda left, right: pd.merge(left, right, on=['squad'], how='outer', 
                                             suffixes=('', '_right')), data_frames)
    
    # Filter out columns ending in '_right' to avoid duplicates after merge.
    mask = df.columns.map(lambda x: 'x_right' not in x)
    df = df[df.columns[mask]]
    
    # Remove duplicate columns, keeping the last occurrence.
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    
    print(f'get_team_data for {end} is loaded')  # Log message indicating completion.
    return df


def get_team_data_vs(top, end, dict_res, type_table='team_vs'):
    """
    Fetches and combines team versus team data across various statistical categories into a single DataFrame.
    
    Parameters:
    - top: Base URL part before category specification.
    - end: URL part after the category, typically parameters or filters.
    - dict_res: A dictionary specifying the features to retrieve for each category.
    - type_table: Specifies the type of table data to fetch, defaulting to 'team_vs'.
    
    Returns:
    - A pandas DataFrame containing merged data across specified categories for team versus team comparisons.
    """
    
    # Fetch data for each category specified in dict_res[type_table]
    # and store each resulting DataFrame in a list.
    data_frames = [
        frame_for_category_team_vs(category, top, end, dict_res[type_table][category])
        for category in ['stats', 'keepers', 'keepersadv', 'shooting', 'passing',
                         'passing_types', 'gca', 'defense', 'possession', 'misc', 'playingtime']
    ]
    
    # Merge all DataFrames on the 'squad' column, resolving any suffix conflicts by removing '_right'.
    df = reduce(lambda left, right: pd.merge(left, right, on=['squad'], how='outer', 
                                             suffixes=('', '_right')), data_frames)
    
    # Filter out columns ending in '_right' to avoid duplicates after merge.
    mask = df.columns.map(lambda x: 'x_right' not in x)
    df = df[df.columns[mask]]
    
    # Remove duplicate columns, keeping the last occurrence.
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    
    print(f'get_team_data_vs for {end} is loaded')  # Log message indicating completion.
    return df

