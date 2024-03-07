import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import dataframe_image as dfi
import numpy as np
from radar_plot import *
from utils import *


def get_df_metrics(df_team, df_team_old, team1, team2, metric_dict, metric_to_plot):
    """
    Generates a DataFrame containing specific metrics for two teams, comparing current and previous data.

    Parameters:
    - df_team (pd.DataFrame): Current team data.
    - df_team_old (pd.DataFrame): Historical team data.
    - team1, team2 (str): Names of the two teams to compare.
    - metric_dict (dict): Mapping of metric names to their respective columns and types.
    - metric_to_plot (list): List of metric names to include in the output DataFrame.

    Returns:
    - pd.DataFrame: Combined metrics for the specified teams, including games played and historical percentiles.
    """

    # Prepare the initial DataFrame for results
    df_metrics = pd.DataFrame(index=['games'] + metric_to_plot)
    
    # Process metrics for each team
    for team in [team1, team2]:
        # Adjust team squad naming directly without copying the DataFrame
        df_filtered = df_team[df_team['squad'].str.replace('vs ', '') == team]

        # Extract games played and other metrics
        games_played = df_filtered['games'].values[0]
        df_metrics[f'Statistic_{team}'] = [games_played] + [df_filtered[metric_dict[m]['m']].values[0] for m in metric_to_plot]

    # Calculate historical percentiles for metrics
    for m_name in metric_to_plot:
        metric_info = metric_dict[m_name]
        historical_values = df_team_old[metric_info['m']]

        # Calculate percentiles and adjust if necessary (type 0 indicates lower is better)
        df_metrics.loc[m_name, 'p5'], df_metrics.loc[m_name, 'p95'] = \
            np.percentile(historical_values, [5, 95] if metric_info['type'] == 1 else [95, 5])

    # Calculate percentile rankings for each team against historical data
    for team in [team1, team2]:
        df_metrics[f'{team}_p'] = df_metrics.apply(
            lambda row: get_perc_by_metric(row[f'Statistic_{team}'], df_team_old[metric_dict[row.name]['m']],
                                           metric_dict[row.name]['type']) if row.name != 'games' else None, axis=1)
    
    return df_metrics


def preprocessing_df_metric(df_metrics, df_team_now, radar_type, radar_order=None):
    """
    Preprocesses the metrics DataFrame for visualization by reformatting and reordering based on specified radar types and orders.

    Parameters:
    - df_metrics (pd.DataFrame): DataFrame containing metrics to be processed.
    - df_team_now (pd.DataFrame): DataFrame containing current team information, used to extract league, season, and games played.
    - radar_type (str): Specifies the type of radar chart to be generated ('Attacking' or 'Defending').
    - radar_order (list, optional): Specifies the order of teams to be presented in the radar chart. Defaults to None.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame ready for radar chart visualization.
    """
    df_ = df_metrics.copy()

    # Extract the team names from the DataFrame's columns
    team1 = df_.columns[0].split('_')[1]
    team2 = df_.columns[1].split('_')[1]

    # Retrieve the league names for each team from df_team_now
    league1 = df_team_now[df_team_now['squad'] == team1]['league_name'].values[0]
    league2 = df_team_now[df_team_now['squad'] == team2]['league_name'].values[0]

    # Get the number of games played by each team and convert to string
    games1 = str(round(df_team_now[df_team_now['squad'] == team1]['games'].values[0]))
    games2 = str(round(df_team_now[df_team_now['squad'] == team2]['games'].values[0]))

    # Get the unique season from df_team_now
    season = df_team_now['season'].unique()[0]

    # Set the new column names for df_
    df_.columns = ['value1', 'value2', 'p5', 'p95', 'perc1', 'perc2']

    # Select and order the subset of columns we are interested in
    df_ = df_[['value1', 'perc1', 'value2', 'perc2']]

    # Drop the first row which may contain header or unwanted info
    df_ = df_.iloc[1:]

    # Create a MultiIndex using team and game information
    y = [(team1, games1, league1, season, 'value'), (team1, games1, league1, season, 'perc'),
         (team2, games2, league2, season, 'value'), (team2, games2, league2, season, 'perc')]
    
    df_.columns = pd.MultiIndex.from_tuples(y)
    df_.columns.names = ['team', '90s played', 'competition', 'season', '']

    # Prepare column identifiers for reordering
    col1 = (team1, games1, league1, season, 'value')
    col2 = (team1, games1, league1, season, 'perc')
    col3 = (team2, games2, league2, season, 'value')
    col4 = (team2, games2, league2, season, 'perc')

    # Check if a specific order for teams is required
    if radar_order is not None:
        # If the order needs to be inverted
        if team1 != radar_order[0]:
            print('team1 invert')
            # Reorder the columns so that team2's data comes first
            df_ = df_[[col3, col4, col1, col2]]
            # Swap team names and their associated games
            team1, team2 = team2, team1
            games1, games2 = games2, games1
    else:
        # Apply different logic based on the radar type
        if radar_type == 'Defending Radar':
            # If team1's percentile is lower than team2's more than half the time, reorder the columns
            if (df_[col2] < df_[col4]).sum() > df_[col1].shape[0] / 2:
                df_ = df_[[col1, col2, col3, col4]]
            else:
                df_ = df_[[col3, col4, col1, col2]]
                # Swap team names and their associated games
                team1, team2 = team2, team1
                games1, games2 = games2, games1
        elif radar_type == 'Attacking Radar':
            # If team1's percentile is higher than team2's more than half the time, reorder the columns
            if (df_[col2] > df_[col4]).sum() > df_[col1].shape[0] / 2:
                df_ = df_[[col1, col2, col3, col4]]
            else:
                df_ = df_[[col3, col4, col1, col2]]
                # Swap team names and their associated games
                team1, team2 = team2, team1
                games1, games2 = games2, games1

    df_ = df_.T.reset_index().set_index('team').T

    df_.columns = [team1, '1', team2, '2']
    df_.loc['90s played'] = [games1, '', games2, '']
    df_.loc['competition'] = [league1, '', league2, '']
    df_.loc['season'] = [season, '', season, '']

    df_.columns.name = 'index_name' 

    # Return the preprocessed DataFrame
    return df_


def metrics_to_image(df_metrics_processed, color_dict, radar_type, date):
    """
    Generate and export an image of a styled DataFrame representing radar chart data for two teams.
    
    Parameters:
    - df_metrics_processed (pd.DataFrame): A preprocessed DataFrame containing metrics data.
    - color_dict (dict): A dictionary mapping team names to their respective colors.
    - radar_type (str): The type of radar chart (e.g., 'Attacking', 'Defending').
    - date (str): The date for which the radar image is being generated, used in file naming.
    
    Returns:
    - An image file generated from the styled DataFrame.
    """
    
    # Copy the processed metrics DataFrame to avoid altering the original data
    df_ = df_metrics_processed.copy()

    # Extract team names and season from the DataFrame
    team1 = df_.columns[0]
    team2 = df_.columns[2]
    season = df_.loc['season'][0]

    # Define the subset of DataFrame to apply styles
    indexes = df_.index[4:]
    columns = ['1', '2']


    # Determine team colors based on availability in the color_dict
    color1, color2 = determine_team_colors(team1, team2, color_dict)

    # Apply styling to the DataFrame
    styler1 = style_dataframe(df_, indexes, columns, color1, color2)
    
    # Create the file name and path for saving the radar image
    rt = '_'.join(radar_type.split(' '))
    FILE_NAME = f'Stats_{rt}_{team1}_{team2}_{date}_{season}.jpeg'
    PATH = f'../img/{date}/stats_image/'

    # Export the styled DataFrame as an image
    dfi.export(styler1, PATH + FILE_NAME, dpi=600)

    # Load and return the image
    image = plt.imread(PATH + FILE_NAME)
    return image

def determine_team_colors(team1, team2, color_dict):
    """Determine colors for the two teams based on the provided color dictionary."""
    if team1 in color_dict:
        color1 = color_dict[team1]
        color2 = '#DB0030' if color1 == '#004D98' else '#004D98'
    elif team2 in color_dict:
        color2 = color_dict[team2]
        color1 = '#DB0030' if color2 == '#004D98' else '#004D98'
    else:
        color1, color2 = '#DB0030', '#004D98'
    return color1, color2

def style_dataframe(df, indexes, columns, color1, color2):
    """Apply styling and format adjustments to the DataFrame based on specified parameters."""
    format_dict = {}

    for n, i in enumerate(df.columns):
        if n % 2 == 0:
            format_dict[i] = '{:.2f}'
        else:
            format_dict[i] = '{:.0f}'
            
        styler = df.style.apply(lambda x: ["color: red" if (v > 90) else \
                      ( "color: blue" if (v < 10) else "") for v in x], 
                subset = (indexes, columns)) \
                    .format(format_dict, subset = (indexes , ))\
                    .set_table_styles(
                   [{
                       'selector': 'th.col_heading, td',
                       'props': [('text-align', 'left')] },
                    {
                       'selector': 'th.col_heading, th.index_name',
                       'props': [('background-color','lightgray')] },
                    {
                       'selector': """th.col_heading.level0.col1, 
                                       th.col_heading.level0.col3,
                                       th.index_name.level0""",
                        'props': [('color', 'lightgray')] }  ,
                    {
                       'selector': """th.col_heading.level0.col0""",
                       'props': [('color', f'{color1}')] }  ,
                                           {
                       'selector': """th.col_heading.level0.col2""",
                       'props': [('color', f'{color2}')] }  ,
                    {
                       'selector': """td.data.row3.col0,
                                      td.data.row3.col1,
                                      td.data.row3.col2,
                                      td.data.row3.col3""",
                       'props': [('font-weight', 'bold')] },
                    ])
            
    return styler


def plot_radar(df_metrics, df_metrics2, radar_type, color_dict, date, image, radar_order=None):
    """
    Generates and displays a radar chart for comparing two teams based on their performance metrics,
    and displays an image with percentiles and statistcs  alongside the radar chart. The function also saves the generated figure.

    Parameters:
    - df_metrics (pd.DataFrame): DataFrame containing metrics for the radar chart calculation.
    - df_metrics2 (pd.DataFrame): Another DataFrame containing metrics, prepared for visualization.
    - radar_type (str): Type of the radar chart to be generated (e.g., "Attacking", "Defending").
    - color_dict (dict): A dictionary mapping team names to colors for the chart.
    - date (str): The date for which the radar is being generated, used in naming the saved file.
    - image (Image): An image to be displayed next to the radar chart.
    - radar_order (list, optional): Specifies the order of teams if a specific order is desired over the default.

    The function checks the radar_order to decide the order of teams for the chart. If not specified,
    it uses the order found in df_metrics2's columns. It then reorganizes data if necessary, calculates ranges
    for the radar chart from provided metrics, and plots the radar chart with matplotlib. Finally,
    it saves the generated figure with a specific naming convention to a designated path.
    """
    # Determine team order based on radar_order or default to the order in df_metrics2
    if radar_order is None:
        team1 = df_metrics2.columns[0]
        team2 = df_metrics2.columns[2]
    else:
        team1, team2 = radar_order
        # Reorder metrics if the specified order doesn't match the default
        if team1 != df_metrics2.columns[0]:
            col1, col2, col3, col4 = df_metrics2.columns[:4]
            df_metrics2 = df_metrics2[[col3, col4, col1, col2]]
            col1, col2 = df_metrics.columns[:2]
            col_ = df_metrics.columns[2:]
            df_metrics = df_metrics[np.concatenate([[col2], [col1], col_])]

    # Extract the season for naming the output file
    season = df_metrics2.loc['season'].values[0]

    # Compute ranges for the radar from the 5th and 95th percentiles
    ranges = df_metrics.iloc[1:, :][['p5', 'p95']].apply(lambda x: (x[0], x[1]), axis=1).tolist()
    # Extract and convert values for plotting
    values1 = df_metrics2.iloc[4:, 0].astype(float).values
    values2 = df_metrics2.iloc[4:, 2].astype(float).values

    # Parameters are the metrics names
    params = df_metrics.iloc[1:, :].index.tolist()

    # Initialize the radar chart with specific design choices
    radar = Radar(label_fontsize=9, range_fontsize=9, fontfamily='serif')

    # Determine the colors for each team
    color1 = color_dict.get(team1, '#DB0030')
    color2 = color_dict.get(team2, '#004D98') if color1 == '#DB0030' else '#DB0030'

    # Setup the title configuration for the radar chart
    title = {
        'title_name': team1,
        'title_color': color1,
        'title_name_2': team2,
        'title_color_2': color2,
        'title_fontsize': 22,
        'subtitle_fontsize': 15,
        'title_description': radar_type,
        'title_description_color': 'gray',
        'title_description_fontsize': 18
    }

    # Placeholder text for the chart's endnote
    endnote = "FOOTSCI"

    # Create figure for plotting
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the radar chart on the first subplot
    radar.plot_radar(ranges=ranges, 
                     params=params, 
                     values=np.stack([values1, values2]), 
                     radar_color=[color1, color2],
                     title=title, 
                     endnote=endnote, dpi=1500,
                     fontfamily='serif', 
                     compare=True,
                     alphas=[0.4, 0.4], 
                     figax=(fig, ax[0]))

    # Display the provided image on the second subplot
    ax[1].imshow(image)

    # Hide axes for the image subplot
    ax[1].axis('off')

    # Construct file name and path for saving the radar chart image
    rt = '_'.join(radar_type.split(' '))
    FILE_NAME = f'{rt}_{team1}_{team2}_{season}.jpeg'
    PATH = f'../img/{date}/radar_image/'

    # Save the figure with high resolution
    fig.savefig(PATH + FILE_NAME, bbox_inches='tight', dpi=400)

    # Show the plot as output
    plt.show()



def get_column_names_from_table(table):
            
    lst = []
    rows = table.find_all('tr')
    
    for row in rows[0].findAll():
        try:
            column_name = row['data-stat']
            lst.append(row['data-stat'])
        except Exception as e:
            pass
    return lst

def get_perc_by_metric(metric_now, df_old, type_):
    
    team_lower_metric = (df_old <= metric_now).sum()
    all_teams = df_old.shape[0]
    if type_ == 1:
        perc = round(team_lower_metric/all_teams*100)
    elif type_ == 0:
        perc = 100 - round(team_lower_metric/all_teams*100)

    
    return perc
