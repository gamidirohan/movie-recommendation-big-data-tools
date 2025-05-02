from data_preprocessing import merge_data
from logger import logger


def create_pivot_table(min_ratings=100):
    """
    Creates a pivot table (users x movies) with ratings.
    Only movies with at least min_ratings are retained.
    """
    data = merge_data()

    # Count number of ratings per movie
    ratings_count = data.groupby("title")["rating"].count()
    popular_movies = ratings_count[ratings_count >= min_ratings].index
    filtered_data = data[data["title"].isin(popular_movies)]

    # Create pivot table: rows = userId, columns = movie title, values = rating
    pivot = filtered_data.pivot_table(index="userId", columns="title", values="rating")
    return pivot


def compute_similarity(pivot, min_periods=20):
    """
    Computes the Pearson correlation matrix between movies.

    Parameters:
    -----------
    pivot : pandas.DataFrame
        The pivot table with users as rows and movies as columns
    min_periods : int, default=20
        Minimum number of common users required to calculate correlation between two movies

    Returns:
    --------
    pandas.DataFrame
        Correlation matrix between movies
    """
    # Use Pearson correlation with a lower minimum number of common users
    # to increase the number of valid correlations
    correlation_matrix = pivot.corr(method="pearson", min_periods=min_periods)

    # Log some statistics about the correlation matrix to help with debugging
    total_cells = correlation_matrix.size
    valid_cells = correlation_matrix.count().sum()
    nan_cells = total_cells - valid_cells
    nan_percentage = (nan_cells / total_cells) * 100 if total_cells > 0 else 0

    logger.info(f"Correlation matrix stats: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]} matrix")
    logger.info(f"Valid correlations: {valid_cells} ({100-nan_percentage:.2f}%), NaN values: {nan_cells} ({nan_percentage:.2f}%)")

    return correlation_matrix


def get_recommendations(movie_title, pivot, correlation_matrix, top_n=10, min_similarity=0.0):
    """
    Returns top_n movie recommendations based on item correlation.

    Parameters:
    -----------
    movie_title : str
        The title of the movie to get recommendations for
    pivot : pandas.DataFrame
        The pivot table with users as rows and movies as columns
    correlation_matrix : pandas.DataFrame
        The correlation matrix between movies
    top_n : int, default=10
        Number of recommendations to return
    min_similarity : float, default=0.0
        Minimum similarity score required for a movie to be recommended

    Returns:
    --------
    pandas.Series
        Series of recommended movies with similarity scores
    """
    if movie_title not in correlation_matrix.columns:
        raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

    # Get the correlation series for the given movie
    similar_movies = correlation_matrix[movie_title].dropna().sort_values(ascending=False)

    # If we have no valid correlations, try a different approach
    if len(similar_movies) <= 1:  # Only the movie itself or empty
        # Log this situation
        logger.warning(f"No valid correlations found for '{movie_title}'. Using a fallback method based on popularity.")

        # Fallback: Use movies with similar genre
        # This requires access to movie genres, which we don't have in this function
        # Instead, we'll return the most popular movies based on number of ratings

        # Count ratings per movie
        ratings_count = pivot.count().sort_values(ascending=False)

        # Exclude the input movie
        if movie_title in ratings_count.index:
            ratings_count = ratings_count.drop(labels=[movie_title])

        # Return top_n most popular movies
        return ratings_count.head(top_n)

    # Remove the movie itself
    if movie_title in similar_movies.index:
        similar_movies = similar_movies.drop(labels=[movie_title])

    # Filter by minimum similarity if specified
    if min_similarity > 0:
        similar_movies = similar_movies[similar_movies >= min_similarity]

    # Return the top_n recommendations
    recommendations = similar_movies.head(top_n)
    return recommendations


if __name__ == "__main__":
    # Test with different min_periods values to see the effect
    pivot = create_pivot_table()

    # Test with original high min_periods (100)
    logger.info("Testing with min_periods=100 (original value)")
    corr_matrix_high = compute_similarity(pivot, min_periods=100)

    # Test with lower min_periods (20)
    logger.info("Testing with min_periods=20 (new default)")
    corr_matrix_low = compute_similarity(pivot, min_periods=20)

    # Test a few different movies
    test_movies = ["Toy Story (1995)", "Star Wars (1977)", "Pulp Fiction (1994)"]

    for movie in test_movies:
        logger.info(f"Testing recommendations for '{movie}'")

        try:
            # Get recommendations with high min_periods
            logger.info("Using high min_periods (100):")
            recs_high = get_recommendations(movie, pivot, corr_matrix_high)
            logger.info(f"Found {len(recs_high)} recommendations")
            logger.info(f"Top 5: {recs_high.head(5)}")
        except Exception as e:
            logger.error(f"Error with high min_periods: {e}")

        try:
            # Get recommendations with low min_periods
            logger.info("Using low min_periods (20):")
            recs_low = get_recommendations(movie, pivot, corr_matrix_low)
            logger.info(f"Found {len(recs_low)} recommendations")
            logger.info(f"Top 5: {recs_low.head(5)}")
        except Exception as e:
            logger.error(f"Error with low min_periods: {e}")

        logger.info("-" * 50)
