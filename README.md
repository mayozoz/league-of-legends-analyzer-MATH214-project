# League of Legends Champion Analysis using Linear Algebra

This project applies linear algebra concepts to analyze champion performance in League of Legends. By representing champion statistics as matrices and vectors, we develop models to predict performance in various game scenarios and optimize champion selection.

## Features

- **Champion Data Collection**: Automatically fetch champion statistics from Riot's Data Dragon API
- **Level Scaling Analysis**: Use linear transformations to model how champion stats scale from level 1 to 18
- **Champion Clustering**: Apply K-means clustering to identify groups of similar champions
- **Principal Component Analysis**: Discover key stat relationships that define champion roles
- **Synergy Calculation**: Measure champion synergies using vector operations
- **Data Visualization**: Generate visualizations of champion statistics and relationships

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Create a `.env` file with your Riot API key:
   ```
   RIOT_API_KEY=your-api-key-here
   VERSION=14.7.1
   ```

## Usage

Run the main analysis application:

```
python lol_analysis_app.py
```

This will present a menu with different analysis options:

1. **View champion statistics**: See top champions by different stats and correlation analysis
2. **Analyze champion scaling**: Visualize how a champion's stats scale from level 1 to 18
3. **Identify champion clusters**: Group champions into clusters based on their stats
4. **Perform PCA analysis**: Identify the key statistical factors that differentiate champions
5. **Calculate champion synergies**: Find which champions work well together based on their stats

## Project Files

- `champion_data_collector.py`: Collects champion data from Riot's API
- `lol_linear_algebra.py`: Core linear algebra functions for champion analysis
- `lol_analysis_app.py`: Interactive application to run various analyses
- `requirements.txt`: List of required Python packages

## Linear Algebra Concepts Used

- **Matrix Representation**: Champions represented as vectors of statistics
- **Linear Transformations**: Modeling level scaling and item effects
- **Vector Operations**: Calculating synergies using dot products and cosine similarity
- **Dimension Reduction**: Using PCA to identify key performance factors
- **Clustering**: Grouping similar champions using vector space representations

## Example Outputs

When running the analysis, various visualizations will be saved:
- Champion PCA: `champion_pca.png`
- Champion Clusters: `champion_clusters.png`
- Stat Correlations: `stat_correlations.png`
- Individual Champion Scaling: `[champion_name]_scaling.png`

## Future Improvements

- Include champion abilities in the analysis
- Add win rate data from match history
- Implement more advanced synergy calculations
- Create a web interface for easier interaction

## License

MIT

## Acknowledgments

This project uses data from Riot Games' Data Dragon API. League of Legends is a trademark of Riot Games, Inc.