import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Arrow, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch, Pitch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from scipy.spatial import distance
import seaborn as sns
import cmasher as cmr  
import unicodedata



def is_inside_box(x,y):
    """
    Check if a point is inside a predefined box.

    Parameters:
    - x (int or float): The x-coordinate of the point.
    - y (int or float): The y-coordinate of the point.

    Returns:
    - bool: True if the point is inside the box, False otherwise.
    """

    if (x >= 20) & (x <= 80) & (y >=82):
        return True
    else:
        return False
    
def update(d, other): d.update(other); return d


def semicircle(r, h, k):
    """
    Calculate the x and y coordinates of a semicircle.

    This function generates the x and y coordinates for a semicircle centered at (h, k)
    with radius r. The semicircle is oriented downwards from the center.

    Parameters:
    - r (float): The radius of the semicircle.
    - h (float): The x-coordinate of the center of the semicircle.
    - k (float): The y-coordinate of the center of the semicircle.

    Returns:
    - tuple: A tuple containing two numpy arrays. The first array contains the x coordinates,
      and the second array contains the y coordinates of the semicircle.
    """
    x0 = h - r  # determine x start
    x1 = h + r  # determine x finish
    x = np.linspace(x0, x1, 10000)  # many points to solve for y

    # use numpy for array solving of the semicircle equation
    y = k - np.sqrt(r**2 - (x - h)**2)  
    return x, y

def sort_key(item):
    # Check if the second element of the tuple is None
    if item[1] is None:
        return float('-inf')  # Return -infinity for None to sort them at the end
    else:
        return item[1][1]  # Return the score for sorting
    
def remove_diacritics(input_str):
    # Normalize the Unicode string to decompose characters with diacritics into two separate characters
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    # Filter out the combining characters (diacritics)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def add_image(ax, img_path, xy, zoom):

    """
    Add an image to a matplotlib Axes object at a specified location and zoom.

    This function reads an image from a given path and places it on a matplotlib
    Axes (`ax`) object at the coordinates specified by `xy`. The image is scaled
    by the `zoom` factor.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to which the image will be added.
    - img_path (str): The file path to the image.
    - xy (tuple): A tuple (x, y) specifying the coordinates where the image's
      center should be placed.
    - zoom (float): The zoom factor for scaling the image. A `zoom` of 1 keeps
      the image at its original size, while a `zoom` of 0.5 halves its size, etc.

    Returns:
    - None
    """
    img = plt.imread(img_path)
    im = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(im, xy, frameon=False, xycoords='data')
    ax.add_artist(ab)


def set_plot_environment():
    plt.rcParams['hatch.linewidth'] = 0.02
    plt.rcParams['font.family'] = 'serif'
    plt.style.use('fivethirtyeight')

def draw_pitch(ax):
    pitch = VerticalPitch(pitch_type='uefa', half=True, goal_type='box', linewidth=1.25, line_color='black', pad_bottom=0, pad_top=20, goal_alpha=0.9, line_zorder=2)
    pitch.draw(ax=ax)
    return pitch


def preprocessing(DF_merged, df_events, index):
    """
    Preprocesses data to filter events and statistics for a specific player.

    This function filters events and statistics from two dataframes based on a specific player
    identified by an index in the merged dataframe. It returns a dictionary containing three dataframes:
    all events for the player, goal events excluding penalties, and player statistics.

    Parameters:
    - DF_merged (pandas.DataFrame): A dataframe containing merged data of players and their statistics.
    - df_events (pandas.DataFrame): A dataframe containing event data for all players.
    - index (int): The index of the specific player in DF_merged to filter data for.

    Returns:
    - dict of pandas.DataFrame: A dictionary containing three filtered dataframes:
      'df_all_events' for all events related to the player, 'df_goals' for goal events by the player
      excluding penalties, and 'df_statistic' for statistics of the player.
    """
    # Extract player ID and team name based on provided index
    player_id = DF_merged['playerId'].iloc[index]
    team_name = DF_merged['teamName'].iloc[index]

    # Filter all events for the specific player and team
    mask = (df_events['playerId'] == player_id) & (df_events['teamName'] == team_name)
    df_all_events = df_events[mask]

    # Further filter for goal events, excluding penalties
    mask = (df_all_events['type'] == 'Goal') & (df_all_events['penaltyScored'] == False)
    df_goals = df_all_events[mask]

    # Filter the player's statistics from the merged dataframe
    mask = DF_merged['playerId'] == player_id
    df_statistic = DF_merged[mask]

    # Compile filtered dataframes into a dictionary
    dict_dfs = {
        'df_all_events': df_all_events,
        'df_goals': df_goals,
        'df_statistic': df_statistic
    }
    
    return dict_dfs

    
def plot_statistics(ax, pitch, df_all_events, df_goals, cmap):
    """
    Plots football match statistics on a given pitch visualization.

    This function uses two datasets: one for all events and another for goal events. It plots
    these events on a provided pitch layout using a hexbin plot for all events and scatter plot
    for goals. The events are differentiated by colors using a colormap for the hexbin plot and
    a specific color for goals.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to plot the statistics on.
    - pitch (mplsoccer.Pitch or similar): The pitch object that provides methods to visualize
      football pitches.
    - df_all_events (pandas.DataFrame): A dataframe containing all event data, with 'x_' and 'y_'
      columns indicating the event locations on the pitch.
    - df_goals (pandas.DataFrame): A dataframe containing goal event data, also with 'x_' and 'y_'
      columns for event locations.
    - cmap (str): The name of the colormap used for plotting the hexbin plot of all events.

    Returns:
    - None
    """
    # Plot all events using a hexbin plot with specified colormap and grid size
    bins1 = pitch.hexbin(x=df_all_events['x_'], y=df_all_events['y_'], ax=ax, cmap=cmap, gridsize=(12, 12),
                         zorder=-1, edgecolors='#f4f4f4', alpha=0.9, lw=.25)
    
    # Plot goal events using a scatter plot with white color and red edges
    bins2 = pitch.scatter(x=df_goals['x_'], y=df_goals['y_'], ax=ax, zorder=2, edgecolors='red',
                          alpha=0.9, lw=.8, color='white')



def plot_semicircle(ax, df_all_events, center_x, center_y):
    """
    Plots a semicircle representing the average distance from a central point for a set of events.

    This function calculates the mean distance of all events from a specified central point (center_x, center_y)
    and plots a semicircle with this average radius on a given Axes object. The semicircle is centered at
    (34, 105) on the pitch, and an annotation indicating the mean distance in meters is added.

    Parameters:
    - ax (matplotlib.axes.Axes): The matplotlib Axes object on which to plot.
    - df_all_events (pandas.DataFrame): DataFrame containing the 'x_' and 'y_' positions of events.
    - center_x (float): The x-coordinate of the center point from which distances are calculated.
    - center_y (float): The y-coordinate of the center point from which distances are calculated.

    Returns:
    - None
    """
    # Calculate mean distance of all events from the central point
    mean = round(df_all_events[['x_', 'y_']].apply(
        lambda x: np.sqrt((x[0] - center_x)**2 + (x[1] - center_y)**2), axis=1).mean(), 1)
    
    # Generate x and y coordinates for a semicircle with the calculated mean radius
    x_circle, y_circle = semicircle(mean, 34, 105)  # Assumes semicircle function is defined elsewhere
    
    # Plot the semicircle on the given Axes object
    ax.plot(x_circle, y_circle, ls='--', color='red', lw=1.5, alpha=1)
    
    # Add an annotation indicating the mean distance
    ax.annotate(
        f"{mean} meters",
        xy=(30, 109),
        xytext=(x_circle[-1] - 7, 109),
        textcoords="data",
        size=8,
        color='red',
        ha='right',
        va='center',
        arrowprops=dict(arrowstyle='<|-, head_width=0.35, head_length=0.65', color='red', lw=0.75)
    )



def add_statistics(ax, player_info, type_):
    """
    Adds statistical annotations to a matplotlib Axes object based on player information and a specified type.

    This function places hexagonal annotations on the plot to represent different player statistics,
    such as shots, non-penalty goals (npG), non-penalty expected goals (npxG), etc. The specific
    statistics displayed depend on the 'type_' parameter.

    Parameters:
    - ax (matplotlib.axes.Axes): The matplotlib Axes object on which to add the annotations.
    - player_info (dict): A dictionary containing player statistics.
    - type_ (str): A string specifying the type of statistics to display. Valid options are
      'Shots', 'np Goals', and 'npxG'. Depending on the choice, different statistics are
      highlighted and ordered differently.

    Returns:
    - None
    """
    # Calculate positions for annotations based on a predefined pattern
    annot_x = [60 - x*13 for x in range(5)]
    
    # Define which statistics to display based on the selected type
    if type_ == 'Shots':
        annot_texts = ['S', 'SoT%', 'npG', 'npxG', 'npxG/S']
    elif type_ == 'np Goals':
        annot_texts = ['npG', 'npxG', 'S', 'SoT%', 'npxG/S']
    elif type_ == 'npxG':
        annot_texts = ['npxG', 'npG', 'S', 'SoT%', 'npxG/S']
    
    # Extract the specified statistics from the player information
    annot_stats = [player_info[stat] for stat in annot_texts]
    
    # Loop through the statistics and add hexagonal annotations for each
    for n, (x, s, stat) in enumerate(zip(annot_x, annot_texts, annot_stats)):
        # Determine the facecolor based on the annotation's index
        facecolor = 'green' if n == 0 else 'None'
        alpha = 0.2 if n == 0 else 1
        
        # Create and add the hexagon patch
        hex_annotation = RegularPolygon(
            (x, 70), numVertices=6, radius=4.5, edgecolor='black',
            facecolor=facecolor, alpha=alpha, hatch='.........', lw=1.25)
        ax.add_patch(hex_annotation)
        
        # Add the statistic abbreviation as an annotation
        ax.annotate(
            text=s, xy=(x, 70), xytext=(0, -18), textcoords='offset points',
            size=7, ha='center', va='center', weight='bold')
        
        # Add the statistic value as another annotation
        text_stat = ax.annotate(
            text=str(stat), xy=(x, 70), xytext=(0, 0), textcoords='offset points',
            size=7, ha='center', va='center', weight='bold')
        text_stat.set_path_effects(
            [path_effects.Stroke(linewidth=1.5, foreground='#efe9e6'), path_effects.Normal()])


def add_annotations(ax, df_all_events, player_info, colorlist):
    """
    Add descriptive annotations to a matplotlib Axes object.

    This function annotates the plot with the player's name, team information, and other relevant statistics.
    It also includes a special annotation for the "SPA" statistic with an arrow and percentage.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to annotate.
    - df_all_events (pandas.DataFrame): DataFrame containing all events, used to get player name.
    - player_info (dict): A dictionary containing player information and statistics.
    - colorlist (list): A list of colors used for annotation texts and arrow properties.

    Returns:
    - None
    """
    # Annotate average distance, player name, and various statistics
    ax.annotate(text="avg distance", xy=(30, 109), xytext=(4, 0), textcoords='offset points', size=8, color='red', ha='left', va='center', alpha=1)
    ax.annotate(text=f"{df_all_events['playerName'].iloc[0].upper()}", xy=(34, 121), size=9, color='black', ha='center', va='center', weight='bold')
    ax.annotate(text=f"{player_info['team']} | {player_info['games']} games | {player_info['minutes_90s']} 90s", xy=(34, 117), size=7, color='black', ha='center', va='center')
    ax.annotate(text=f"{player_info['shots']} shots | {player_info['npxg_all']} npxG | {player_info['npgoals']} npG", xy=(34, 113), size=7, color='black', ha='center', va='center')
    
    # Special annotation for SPA statistic
    share = int(player_info['share'] * 100)
    ax.annotate(text=f"{share}%", xy=(18, 100), xytext=(7, 100), size=8, color=colorlist[-2], ha='center', va='center', weight='bold', arrowprops=dict(arrowstyle='-|>', color=colorlist[-2], lw=1.5))
    ax.annotate(text="SPA", xy=(7, 97), size=7, color=colorlist[-2], ha='center', va='center', weight='bold')

 
def add_title(fig, ax, player_info, league, date, min_minutes_90s, colorlist, type_):
    """
    Adds titles, subtitles, and custom annotations to a matplotlib figure.

    This function decorates a matplotlib figure with comprehensive titles, subtitles, and various
    annotations, including arrows and polygons, to represent statistical data visually. It is tailored
    for visualizations related to football analytics.

    Parameters:
    - fig (matplotlib.figure.Figure): The figure object to add titles and annotations to.
    - ax (matplotlib.axes.Axes): The Axes object for the figure (not directly used in this function).
    - player_info (dict): Dictionary containing player information (not directly used in this function).
    - league (str): Name of the league to be included in the title.
    - date (str): Date for the subtitle.
    - min_minutes_90s (int or float): Minimum minutes played, normalized to 90-minute matches, for inclusion in the subtitle.
    - colorlist (list): A list of color hex codes for styling the annotations.
    - type_ (str): The type of statistic being visualized, included in the title.

    Returns:
    - None
    """
    font = 'serif'
    
    # Main title and subtitles using fig.text instead of fig_text
    fig.text(x=0.51, y=0.8, s=f"TOP9 in {league} by {type_} p90", va="bottom", ha="center",
             fontsize=18, color="black", font=font, weight="bold")
    fig.text(x=0.51, y=0.79, s=f"Non-penalty shot bins for {league} players and per game stat | Season 2023/2024 | {date} | viz by @MBorodastov",
             va="bottom", ha="center", fontsize=8, font=font)
    fig.text(x=0.51, y=0.78, s=f"{min_minutes_90s}+ 90s minutes", va="bottom", ha="center",
             fontsize=8, font=font)
    fig.text(x=0.51, y=0.77, s="Data: Opta", va="bottom", ha="center", fontsize=8, font=font)
    
    # A custom annotation in the corner of the figure
    fig.text(x=0.87, y=0.15, s="FOOTSCI", color=colorlist[-2], weight='bold', va="bottom",
             ha="center", fontsize=18, font=font)

    # Adding hexagons to represent some data visually
    annot_x = [1130 + i for i in range(70, 491, 70)]
    y_height = 2955
    for x, color in zip(annot_x, colorlist[2:]):
        hex_annotation = RegularPolygon((x, y_height), numVertices=6, radius=30, edgecolor='#f4f4f4',
                                        fc=color, lw=0.5)
        fig.patches.extend([hex_annotation])

    # Arrows and a circle for visual cues
    arrow1 = Arrow(x=1530, y=y_height, dx=100, dy=0, width=20, color=colorlist[-2])
    arrow2 = Arrow(x=1150, y=y_height, dx=-100, dy=0, width=20, color=colorlist[-2])
    circle = Circle(xy=(1900, y_height), radius=12, facecolor='white', edgecolor='red')
    fig.patches.extend([arrow1, arrow2, circle])

    # Additional text annotations at the bottom of the figure
    fig.text(x=0.325, y=0.756, s="Shot frequency:   lower", weight='bold', va="bottom",
             ha="center", fontsize=8, font=font)
    fig.text(x=0.64, y=0.756, s="higher", weight='bold', va="bottom", ha="center", fontsize=8, font=font)
    fig.text(x=0.74, y=0.756, s="np Goals", weight='bold', va="bottom", ha="center", fontsize=8, font=font)
