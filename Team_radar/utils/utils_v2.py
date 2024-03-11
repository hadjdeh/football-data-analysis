import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import dataframe_image as dfi
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.patches as mpatches
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
            league1, league2 = league2, league1

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
                league1, league2 = league2, league1

        elif radar_type == 'Attacking Radar':
            # If team1's percentile is higher than team2's more than half the time, reorder the columns
            if (df_[col2] > df_[col4]).sum() > df_[col1].shape[0] / 2:
                df_ = df_[[col1, col2, col3, col4]]
            else:
                df_ = df_[[col3, col4, col1, col2]]
                # Swap team names and their associated games
                team1, team2 = team2, team1
                games1, games2 = games2, games1
                league1, league2 = league2, league1

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


def plot_radar(df_metrics, df_metrics2, radar_type, color_dict, date, image, radar_order=None, team_name=None):
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
    color1, color2 = determine_team_colors(team1, team2, color_dict)


    if team_name != None:
        competition = df_metrics2[team_name]['competition']
        games = df_metrics2[team_name]['90s played']
        values = df_metrics2[team_name].iloc[4:]
        
        colormap = mpl.cm.get_cmap('RdYlBu')
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        # color1 = colormap(norm(0.1))
        # color2 = colormap(norm(0.9))
        color1 = color_dict[team_name][0]
        color2 = color_dict[team_name][1]
        subtitle = competition + '\n' + season + '\n' + games + ' games'+ '\n' + date

    # Setup the title configuration for the radar chart
    title = {
        'title_name': team1 if team_name is None else team_name,
        'title_color': color1,
        'title_name_2': team2 if team_name is None else '',
        'subtitle_name': '' if team_name is None else subtitle,
        'title_color_2': color2,
        'title_fontsize': 21,
        'subtitle_fontsize': 15,
        'title_description': radar_type,
        'title_description_color': 'gray' if team_name is None else 'black',
        'title_description_fontsize': 18
    }

    # Placeholder text for the chart's endnote
    endnote = "FOOTSCI"

    # Create figure for plotting
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the radar chart on the first subplot
    radar.plot_radar(ranges=ranges, 
                     params=params, 
                     values=np.stack([values1, values2]) if team_name is None else values, 
                     radar_color=[color1, color2],
                     title=title, 
                     endnote=endnote, dpi=1500,
                     fontfamily='serif', 
                     compare=True if team_name is None else False,
                     alphas=[0.4, 0.4], 
                     figax=(fig, ax[0]))

    # Display the provided image on the second subplot
    ax[1].imshow(image)

    # Hide axes for the image subplot
    ax[1].axis('off')

    # Construct file name and path for saving the radar chart image
    rt = '_'.join(radar_type.split(' '))
    FILE_NAME = f'{rt}_{team1}_{team2}_{season}.jpeg' if team_name is None else f'{rt}_dist_{team_name}_{season}.jpeg'
    PATH = f'../img/{date}/radar_image/'

    # Save the figure with high resolution
    fig.savefig(PATH + FILE_NAME, bbox_inches='tight', dpi=400)

    # Show the plot as output
    plt.show()



def get_dist(df_old, df_now, df_metrics, team_name, metrics_dict, radar_type, date):
    """
    Plots distribution graphs for a specified team's performance metrics.

    Parameters:
    - df_old: DataFrame containing old performance metrics.
    - df_now: DataFrame containing current performance metrics (unused in current implementation).
    - df_metrics: DataFrame with specific metrics to be visualized.
    - team_name: Name of the team for which the metrics are plotted.
    - metrics_dict: Dictionary mapping metric names to their aliases.
    """
    # Customize the appearance of the plots
    mpl.rcParams['axes.spines.left'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.bottom'] = True
    
    # Select the metrics to be plotted
    df_metrics_ = df_metrics.iloc[4:]
    n_axes = df_metrics_.shape[0]
    
    season = df_metrics.loc['season'].values[0]
    
    # Create subplots for each metric
    fig, ax = plt.subplots(n_axes,1, figsize=(20,n_axes *2.1 ))
    
    
    for num, metric_name in enumerate(df_metrics_.index):
        
        series = df_old[metrics_dict[metric_name]['m']]
        metric_value = df_metrics_[team_name][metric_name]
        metric_type = metrics_dict[metric_name]['type']
        
        # Calculate percentile values and the minimum and maximum of the series
        metric_p_5 = round(np.percentile(series,5),2)
        metric_p_95 = round(np.percentile(series,95),2)
        metric_min = round(series.min(),2)
        metric_max = round(series.max(),2)
        perc_metric = get_perc_by_metric(metric_value, series, metric_type)
        
        
        # Set the color map based on the metric type
        colourmap = mpl.cm.get_cmap('RdYlBu').reversed() if metric_type == 1 else mpl.cm.get_cmap('RdYlBu')
        
    
        if metric_min >= 0:
            ax[num].set_xlim(0, metric_max*1.15)
        else:
            print('neg')
            ax[num].set_xlim(metric_min*1.1, metric_max*1.4)
            
        if metric_type == 0:
            
            ax[num].invert_xaxis()
            metric_p_5, metric_p_95 = metric_p_95, metric_p_5
            
            if metric_min >= 0:
                ax[num].set_xlim(metric_max*1.15, 0)
            else:
                print('neg')
                ax[num].set_xlim(metric_min*1.1, metric_max*1.4)
                
                
        # Plot the distribution of the metric
        sns.kdeplot(series, bw=0.45, color='lightgray', ax=ax[num],gridsize=400)

        # Highlight the metric value with an arrow
        y_lim_max = ax[num].get_ylim()[1]
        ax[num].annotate("", xy=(metric_value, 0), xytext=(metric_value, y_lim_max*0.04),
                     arrowprops=dict(arrowstyle="<|-", color='blue',linewidth=3), 
                     fontsize=15)        
        
        
        # Color the distribution based on percentile values
        kde_x, kde_y = ax[num].lines[0].get_data()

        vmin = metric_p_5 if metric_p_5 < metric_p_95 else metric_p_95
        vmax = metric_p_95 if metric_p_5 < metric_p_95 else metric_p_5
        normalize = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        npts = 400
        for i in range(npts - 1):
            ax[num].fill_between([kde_x[i], kde_x[i+1]],
                                 [kde_y[i], kde_y[i+1]],
                                  color=colourmap(normalize(kde_x[i])),
                                  alpha=1)
            ax[num].fill_between(kde_x, kde_y, 
                                 where=(kde_x>metric_value if metric_type == 1 else kde_x<metric_value),
                                 color='lightgray')
            
        # Add a legend to the plot
        ax[num].legend([
#                     '',
            f'{round(metric_value,2)}',
            f'P {perc_metric}',
       ], fontsize=15, loc=4)

        legend_entries = [
        mpatches.Patch(color='none', label=f'{round(metric_value,2)}'),  
        mpatches.Patch(color='none', label=f'P{perc_metric}')            
        ]   

        leg = ax[num].legend(handles=legend_entries, fontsize=15, loc=4, ncol=2, 
              handlelength=0, handletextpad=-1.5, frameon=False)
    
        text1 = leg.texts[1]
        color = colourmap(normalize(metric_value))
#         color = tuple([i+0.4  if (n == 2) & (i<0.3) else i for n, i in enumerate(color)])
        text1.set_color(color)
        text1.set_weight('bold')

        text2 = leg.texts[0]
        text2.set_color('blue')
        text2.set_weight('bold')
    
        ax[num].set_xlabel('')
        ax[num].set_yticks([])
        ax[num].set_xticks([])
        metric_name_ = metric_name.replace('C.','')
        ax[num].set_ylabel(metric_name_ if len(metric_name_.split(' ',1)) == 1 else 
                           metric_name_.split(' ',1)[0] + '\n' + metric_name_.split(' ',1)[1], 
                           fontweight='semibold',fontfamily='serif',fontsize=20, rotation=0)
        ax[num].yaxis.set_label_coords(-0.05,0.3)
    
    
        round_ = 2 if abs(metric_max - metric_min) < 2 else 1

        if metric_type == 1:
            if metric_min >= 0:
                x_lim_min = 0
                x_lim_max = round(metric_max,round_)
            else:
                x_lim_min = metric_min
                x_lim_max = round(metric_max,round_)
        elif metric_type == 0:
            if metric_min >= 0:
                x_lim_min = round(metric_max,round_)
                x_lim_max = 0
            else:
                x_lim_min = metric_min
                x_lim_max = round(metric_max,round_)   
                

        ax[num].axvline(round(metric_p_5,round_), ls='--', c='gray')
        ax[num].axvline(round(metric_p_95,round_), ls='--', c='gray')

        ax[num].set_xticks([x_lim_min, round(metric_p_5,round_), round(metric_p_95,round_), x_lim_max])
        ax[num].tick_params(axis='x', labelsize=14, pad=5)
        
    plt.subplots_adjust(hspace=0.5)
    
    ax[0].set_title('Distributions', fontsize=40, fontweight='bold', pad=60,
                   fontfamily='serif', color='black')
    
    
    # Create the file name and path for saving the distri image
    rt = '_'.join(radar_type.split(' '))
    FILE_NAME = f'Distri_{rt}_{team_name}_{date}_{season}.jpeg'
    PATH = f'../img/{date}/distri_image/'

    fig.savefig(PATH + FILE_NAME, bbox_inches='tight', dpi=400)
    plt.close(fig)

    # Load and return the image
    image = plt.imread(PATH + FILE_NAME)
    return image




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
