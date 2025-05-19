import matplotlib.pyplot as plt
import os

def plot_with_markers(df):
    """
    Genereert een zwart-wit lijnplot met markers voor elke kolom in een DataFrame.

    Parameters:
    - df: pandas.DataFrame met waarden (index = x-as, kolommen = lijnen)
    - title: titel van de plot
    - xlabel: label x-as
    - ylabel: label y-as
    - filename: bestandsnaam om op te slaan (zonder extensie), indien None: toont de plot enkel
    - legend_title: optioneel, titel van de legenda
    """
 

    markers = ['o', 's', 'x', 'D', '^', 'v', '*', '+', '1', '2', '3', '4']
    plt.figure()

    for i, col in enumerate(df.columns):
        plt.plot(df.index, df[col], marker=markers[i % len(markers)], linestyle='-', label=col)


def save_plot(dir, name):
    os.makedirs(f"C:/Users/rikte/VS Code Python Projects/thesis_riktgs/figures/{dir}", exist_ok=True)
    path = f"C:/Users/rikte/VS Code Python Projects/thesis_riktgs/figures/{dir}/{name}.png"
    # plt.figure(figsize=(6.4, 4.8))
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')