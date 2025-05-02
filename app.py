import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import random
import plotly.express as px
import plotly.graph_objects as go
from advanced_recommender import create_pivot_table, advanced_recommendations
from recommendation_engine import (
    create_pivot_table as create_pivot_table_traditional,
    get_recommendations,
    compute_similarity,
)
from data_preprocessing import load_movies, load_movies_with_genres
from logger import logger
from sklearn.metrics.pairwise import cosine_similarity


# Set page configuration
st.set_page_config(
    page_title="Enhanced Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/dinesh-git17/movie_recommendation',
        'About': 'Movie Recommendation System with enhanced UI and analytics'
    }
)

# Custom CSS for better UI
def load_css(dark_mode=False):
    """Load custom CSS with support for light and dark mode"""

    # Define colors for light and dark modes
    if dark_mode:
        # Dark mode colors
        bg_color = "#121212"
        text_color = "#E0E0E0"
        card_bg = "#1E1E1E"
        primary_color = "#4F6BFF"
        secondary_bg = "#2D2D2D"
        accent_bg = "#2D3748"
        hover_color = "#3B4FD9"
        border_color = "#4F6BFF"
        tab_bg = "#2D2D2D"
        tab_active_bg = "#4F6BFF"
        tab_active_color = "#FFFFFF"
        metric_card_bg = "#2D3748"
        recommendation_header_bg = "#2D3748"
        feedback_form_bg = "#2D2D2D"
    else:
        # Light mode colors
        bg_color = "#f5f7f9"
        text_color = "#1E293B"
        card_bg = "#FFFFFF"
        primary_color = "#1E3A8A"
        secondary_bg = "#EFF6FF"
        accent_bg = "#DBEAFE"
        hover_color = "#2563EB"
        border_color = "#1E3A8A"
        tab_bg = "#FFFFFF"
        tab_active_bg = "#1E3A8A"
        tab_active_color = "#FFFFFF"
        metric_card_bg = "#EFF6FF"
        recommendation_header_bg = "#DBEAFE"
        feedback_form_bg = "#F3F4F6"

    st.markdown(f"""
    <style>
    /* Base styles */
    .main {{
        background-color: {bg_color};
        color: {text_color};
    }}
    /* Add more padding to the main content */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}
    /* Text colors */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, span, div {{
        color: {text_color};
    }}

    /* Main tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {tab_bg};
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 20px;
        padding-right: 20px;
        min-width: 140px;
        text-align: center;
        color: {text_color};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {tab_active_bg};
        color: {tab_active_color};
    }}

    /* Genre tabs styling - for nested tabs */
    .stTabs [data-baseweb="tab-panel"] .stTabs [data-baseweb="tab"] {{
        padding-left: 15px;
        padding-right: 15px;
        min-width: 120px;
        font-size: 0.9em;
    }}

    /* Add padding to tab content */
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }}

    /* Movie card styling */
    .movie-card {{
        background-color: {card_bg};
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        margin-top: 10px;
        border-left: 5px solid {border_color};
        transition: transform 0.2s;
    }}
    .movie-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    }}

    /* Metric card styling */
    .metric-card {{
        background-color: {metric_card_bg};
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }}

    /* Other UI elements */
    .recommendation-header {{
        background-color: {recommendation_header_bg};
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }}
    .feedback-form {{
        background-color: {feedback_form_bg};
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: {hover_color};
    }}

    /* Dark/Light mode toggle styling */
    .mode-toggle {{
        display: flex;
        align-items: center;
        justify-content: flex-end;
        margin-bottom: 10px;
    }}
    .mode-toggle-label {{
        margin-right: 10px;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

# Function to create the dark mode toggle
def create_dark_mode_toggle():
    """Create a toggle switch for dark/light mode"""
    # Check if dark mode is in session state, initialize if not
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

    # Create a container for the toggle
    toggle_container = st.container()

    # Add the toggle in the top-right corner
    with toggle_container:
        col1, col2, col3 = st.columns([1, 6, 1])
        with col3:
            # Create the toggle with sun/moon icons
            if st.session_state.dark_mode:
                icon = "☀️"  # Sun icon for switching to light mode
                label = "Light Mode"
            else:
                icon = "🌙"  # Moon icon for switching to dark mode
                label = "Dark Mode"

            if st.button(f"{icon} {label}"):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()

    return st.session_state.dark_mode

@st.cache_resource
def get_pivot_advanced():
    # Create the pivot table for advanced recommendations.
    logger.info("Generating advanced pivot table...")
    pivot = create_pivot_table()
    logger.info("Pivot table generated successfully.")
    return pivot


@st.cache_resource
def get_pivot_traditional():
    # Create the pivot table for traditional recommendations.
    pivot = create_pivot_table_traditional()
    return pivot


@st.cache_resource
def get_movies_with_genres():
    # Load full movie data including genres
    logger.info("Loading movies with genre information...")
    movies = load_movies_with_genres()
    logger.info("Movies with genres loaded successfully.")
    return movies


def create_movie_card(movie_title, genres, year, similarity_score=None):
    """Create a styled card for movie display with image placeholder"""
    # Get the current theme mode
    dark_mode = st.session_state.get('dark_mode', False)

    # Set colors based on theme
    if dark_mode:
        bg_color = "#1E1E1E"
        text_color = "#E0E0E0"
        border_color = "#4F6BFF"
    else:
        bg_color = "white"
        text_color = "#1E293B"
        border_color = "#1E3A8A"

    # Generate a random color for the movie poster placeholder
    poster_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    # Extract first letter of each word for the poster placeholder
    initials = ''.join([word[0] for word in movie_title.split() if word[0].isalpha()])[:2].upper()

    # Create a simple card with minimal HTML
    card_html = f"""
    <div style="background-color: {bg_color}; border-radius: 10px; padding: 25px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 25px; margin-top: 10px;
                border-left: 5px solid {border_color};">
        <div style="display: flex; gap: 20px;">
            <div style="min-width: 90px; height: 130px; background-color: {poster_color};
                        display: flex; align-items: center; justify-content: center;
                        color: white; font-size: 24px; font-weight: bold; border-radius: 5px;">
                {initials}
            </div>
            <div style="padding-top: 5px;">
                <h3 style="margin-top: 0; margin-bottom: 15px; color: {text_color};">{movie_title}</h3>
                <p style="margin-bottom: 10px; color: {text_color};"><strong>Year:</strong> {year}</p>
                <p style="margin-bottom: 10px; color: {text_color};"><strong>Genres:</strong> {genres}</p>"""

    if similarity_score is not None:
        card_html += f"<p style=\"margin-bottom: 10px; color: {text_color};\"><strong>Similarity Score:</strong> {similarity_score:.2f}</p>"

    card_html += """
            </div>
        </div>
    </div>
    """
    return card_html


def plot_genre_distribution(movies_df):
    """Create a bar chart showing the distribution of movie genres"""
    # Explode the genres_list column to get one row per genre
    genre_counts = {}
    for genres in movies_df['genres_list']:
        for genre in genres:
            if genre in genre_counts:
                genre_counts[genre] += 1
            else:
                genre_counts[genre] = 1

    # Convert to DataFrame for plotting
    genre_df = pd.DataFrame({
        'Genre': list(genre_counts.keys()),
        'Count': list(genre_counts.values())
    }).sort_values('Count', ascending=False)

    # Create a bar chart with Plotly
    fig = px.bar(
        genre_df,
        x='Genre',
        y='Count',
        title='Distribution of Movie Genres',
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_title='Genre',
        yaxis_title='Number of Movies',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    return fig


def plot_rating_heatmap(pivot):
    """Create a heatmap of movie ratings"""
    # Sample a subset of the pivot table for visualization
    sample_size = min(50, pivot.shape[0])
    sample_pivot = pivot.sample(sample_size).iloc[:, :50]

    # Create a heatmap with Plotly
    fig = px.imshow(
        sample_pivot.fillna(0),
        color_continuous_scale='Blues',
        title='Movie Rating Heatmap (Sample)',
        labels=dict(x="Movies", y="Users", color="Rating")
    )
    fig.update_layout(height=600)
    return fig


def plot_recommendation_comparison(movie_title, trad_recs, adv_recs):
    """Create a comparison chart for traditional vs advanced recommendations"""
    # Create a simple bar chart showing top 5 recommendations from each method
    fig = go.Figure()

    # Process traditional recommendations
    trad_top5 = trad_recs.head(5)
    trad_movies = trad_top5.index.tolist()
    trad_scores = trad_top5.values.tolist()

    # Process advanced recommendations
    adv_top5 = adv_recs.head(5)
    adv_movies = adv_top5.index.tolist()
    adv_scores = adv_top5.values.tolist()

    # Add traditional recommendations
    fig.add_trace(go.Bar(
        x=[f"Trad: {movie[:20]}..." if len(movie) > 20 else f"Trad: {movie}" for movie in trad_movies],
        y=trad_scores,
        name='Traditional',
        marker_color='#1E3A8A'
    ))

    # Add advanced recommendations
    fig.add_trace(go.Bar(
        x=[f"Adv: {movie[:20]}..." if len(movie) > 20 else f"Adv: {movie}" for movie in adv_movies],
        y=adv_scores,
        name='Advanced',
        marker_color='#3B82F6'
    ))

    fig.update_layout(
        title=f'Top 5 Recommendations for "{movie_title}"',
        xaxis_title='Movie',
        yaxis_title='Score',
        barmode='group',
        height=500,
        xaxis={
            'tickangle': -45,
            'tickfont': {'size': 10}
        }
    )

    return fig


def plot_feedback_trends(feedback_file="feedback.csv"):
    if os.path.exists(feedback_file):
        feedback = pd.read_csv(feedback_file)
        st.subheader("Average Rating per Recommended Movie")
        # Plot average rating per recommended movie
        avg_ratings = (
            feedback.groupby("recommended_movie")["user_rating"].mean().reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x="recommended_movie",
            y="user_rating",
            data=avg_ratings,
            ax=ax,
            color="skyblue",
        )
        ax.set_ylabel("Average Rating")
        ax.set_xlabel("Recommended Movie")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

        st.subheader("Feedback Count Over Time")
        # Plot feedback count over time
        feedback["timestamp"] = pd.to_datetime(feedback["timestamp"])
        feedback["date"] = feedback["timestamp"].dt.date
        daily_counts = feedback.groupby("date").size().reset_index(name="count")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x="date", y="count", data=daily_counts, marker="o", ax=ax2)
        ax2.set_ylabel("Feedback Count")
        ax2.set_xlabel("Date")
        st.pyplot(fig2)
    else:
        st.info("No feedback data available yet.")


def main():
    # Add dark mode toggle at the top
    dark_mode = create_dark_mode_toggle()

    # Apply custom CSS with dark mode setting
    load_css(dark_mode=dark_mode)

    # Create sidebar for navigation and filters with improved styling
    with st.sidebar:
        # Logo and title with better styling
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://img.icons8.com/fluency/96/movie.png", width=80)
        with col2:
            # Set title color based on theme
            title_color = "#4F6BFF" if st.session_state.get('dark_mode', False) else "#1E3A8A"
            st.markdown(f"<h2 style='margin-top: 10px; color: {title_color};'>Movie<br>Recommender</h2>", unsafe_allow_html=True)

        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

        # User analytics summary in sidebar
        if os.path.exists("feedback.csv"):
            feedback_df = pd.read_csv("feedback.csv")
            total_ratings = len(feedback_df)
            avg_rating = feedback_df["user_rating"].mean()

            st.markdown("### User Analytics")
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Ratings</h3>
                <h2>{total_ratings}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <h3>Average Rating</h3>
                <h2>{avg_rating:.1f}/5.0</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div style="background-color: #EFF6FF; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <p style="margin: 0;"><strong>No user data available yet.</strong><br>Start rating movies to see analytics!</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Add a genre filter with styled header
        movies_with_genres = get_movies_with_genres()
        all_genres = set()
        for genres in movies_with_genres['genres_list']:
            all_genres.update(genres)

        # Set header color based on theme
        header_color = "#4F6BFF" if st.session_state.get('dark_mode', False) else "#1E3A8A"
        st.markdown(f"<h3 style='color: {header_color}; margin-bottom: 10px;'>Filter Movies</h3>", unsafe_allow_html=True)

        # Set info box background color based on theme
        info_bg_color = "#2D3748" if st.session_state.get('dark_mode', False) else "#DBEAFE"
        info_text_color = "#E0E0E0" if st.session_state.get('dark_mode', False) else "#1E293B"

        st.markdown(f"""
        <div style="background-color: {info_bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 14px; color: {info_text_color};">Select genres and year range to filter movie recommendations</p>
        </div>
        """, unsafe_allow_html=True)

        selected_genres = st.multiselect(
            "Select Genres",
            options=sorted(list(all_genres)),
            default=[]
        )

        # Year range filter
        years = movies_with_genres['year'].unique()
        years = [y for y in years if y != 'Unknown']
        years = sorted([int(y) for y in years])

        year_range = st.slider(
            "Select Year Range",
            min_value=min(years),
            max_value=max(years),
            value=(min(years), max(years))
        )

        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
        # Set header color based on theme
        header_color = "#4F6BFF" if st.session_state.get('dark_mode', False) else "#1E3A8A"
        st.markdown(f"<h3 style='color: {header_color};'>About</h3>", unsafe_allow_html=True)

        # Use a simpler approach with st.container and st.write
        with st.container():
            # Set about box background color based on theme
            about_bg_color = "#2D3748" if st.session_state.get('dark_mode', False) else "#EFF6FF"
            about_text_color = "#E0E0E0" if st.session_state.get('dark_mode', False) else "#1E293B"

            st.markdown(
                f"""
                <div style="background-color: {about_bg_color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <p style="color: {about_text_color};">This enhanced dashboard provides interactive movie recommendations with detailed analytics.</p>
                    <p style="color: {about_text_color};"><b>Features:</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Use native Streamlit components for the list
            st.write("• Advanced recommendation engine")
            st.write("• Interactive data visualization")
            st.write("• Personalized movie suggestions")
            st.write("• A/B testing of algorithms")

    # Main content
    st.title("🎬 Enhanced Movie Recommendation Dashboard")

    # Set info box background color based on theme
    info_bg_color = "#2D3748" if st.session_state.get('dark_mode', False) else "#EFF6FF"
    info_text_color = "#E0E0E0" if st.session_state.get('dark_mode', False) else "#1E293B"

    st.markdown(f"""
    <div style="background-color: {info_bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
    <p style="color: {info_text_color}; margin: 0;">Explore movie recommendations, visualize trends, and discover new films with our enhanced interactive dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs with icons
    tabs = st.tabs([
        "🎯 Recommendations",
        "📊 Analytics",
        "🔍 Movie Explorer",
        "🧪 A/B Testing"
    ])

    # Tab 1: Enhanced Recommendations
    with tabs[0]:
        st.header("Movie Recommendations")

        # Create two columns for layout with better width distribution
        col1, col2 = st.columns([1, 4])

        with col1:
            # Movie selection with genre filtering
            pivot = get_pivot_advanced()
            movie_list = list(pivot.columns)

            # Apply genre filter if selected
            if selected_genres:
                filtered_movies = []
                for movie in movie_list:
                    movie_info = movies_with_genres[movies_with_genres['title'] == movie]
                    if not movie_info.empty:
                        movie_genres = movie_info.iloc[0]['genres_list']
                        if any(genre in movie_genres for genre in selected_genres):
                            filtered_movies.append(movie)
                movie_list = filtered_movies

            # Apply year filter
            if year_range:
                filtered_by_year = []
                for movie in movie_list:
                    movie_info = movies_with_genres[movies_with_genres['title'] == movie]
                    if not movie_info.empty:
                        movie_year = movie_info.iloc[0]['year']
                        if movie_year != 'Unknown':
                            if year_range[0] <= int(movie_year) <= year_range[1]:
                                filtered_by_year.append(movie)
                movie_list = filtered_by_year

            movie_list.sort()

            st.markdown("""
            <div class="recommendation-header">
                <h3>Select a Movie</h3>
                <p>Choose a movie to get personalized recommendations</p>
            </div>
            """, unsafe_allow_html=True)

            if not movie_list:
                st.warning("No movies match your filters. Please adjust your selection.")
                selected_movie = None
            else:
                selected_movie = st.selectbox("Choose a movie", movie_list)

                # Display selected movie details
                if selected_movie:
                    movie_info = movies_with_genres[movies_with_genres['title'] == selected_movie]
                    if not movie_info.empty:
                        movie_data = movie_info.iloc[0]
                        st.markdown(create_movie_card(
                            movie_data['title'],
                            movie_data['genres_str'],
                            movie_data['year']
                        ), unsafe_allow_html=True)

        with col2:
            if selected_movie:
                if st.button("Get Recommendations", key="advanced"):
                    try:
                        logger.info("Generating recommendations for: %s", selected_movie)
                        recommendations = advanced_recommendations(selected_movie, pivot)

                        st.markdown(f"""
                        <div style="background-color: #EFF6FF; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                            <h3>Movies similar to "{selected_movie}"</h3>
                            <p>Based on collaborative filtering and matrix factorization</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Convert to DataFrame for display
                        rec_df = recommendations.reset_index().rename(
                            columns={selected_movie: "Similarity Score"}
                        )

                        # Create a container with some padding for the recommendations
                        with st.container():
                            # Display recommendations as cards
                            for idx, row in rec_df.iterrows():
                                rec_movie = row["index"]
                                score = row["Similarity Score"]

                                movie_info = movies_with_genres[movies_with_genres['title'] == rec_movie]
                                if not movie_info.empty:
                                    movie_data = movie_info.iloc[0]
                                    st.markdown(create_movie_card(
                                        movie_data['title'],
                                        movie_data['genres_str'],
                                        movie_data['year'],
                                        score
                                    ), unsafe_allow_html=True)

                        # Display feedback form using st.form to avoid re-runs on slider change
                        st.markdown("""
                        <div class="feedback-form">
                            <h3>Provide Feedback</h3>
                            <p>Rate the recommendations to help improve future suggestions</p>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.form(key="feedback_form"):
                            feedback = []
                            for idx, row in rec_df.iterrows():
                                rec_movie = row["index"]
                                rating = st.slider(
                                    f"Rate '{rec_movie}':",
                                    min_value=1,
                                    max_value=5,
                                    value=3,
                                    key=f"slider_{idx}",
                                )
                                feedback.append(
                                    {
                                        "selected_movie": selected_movie,
                                        "recommended_movie": rec_movie,
                                        "similarity_score": row["Similarity Score"],
                                        "user_rating": rating,
                                        "timestamp": datetime.datetime.now().isoformat(),
                                    }
                                )
                            submitted = st.form_submit_button("Submit Feedback")
                            if submitted:
                                # Save feedback using an absolute path
                                feedback_file = os.path.join(os.getcwd(), "feedback.csv")
                                logger.info("Saving feedback to: %s", feedback_file)
                                try:
                                    df_feedback = pd.DataFrame(feedback)
                                    header_needed = not os.path.exists(feedback_file)
                                    df_feedback.to_csv(
                                        feedback_file,
                                        mode="a",
                                        header=header_needed,
                                        index=False,
                                    )
                                    st.success("Feedback submitted successfully!")
                                    logger.info(
                                        "Feedback submitted for movie: %s", selected_movie
                                    )
                                except Exception as ex:
                                    st.error(f"Error saving feedback: {ex}")
                                    logger.error("Error saving feedback: %s", ex)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        logger.error("Error generating recommendations: %s", e)

    # Tab 2: Analytics Dashboard
    with tabs[1]:
        st.header("Movie Analytics Dashboard")

        # Create three columns for metrics
        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Movies</h3>
                <h2>{}</h2>
            </div>
            """.format(len(movies_with_genres)), unsafe_allow_html=True)

        with metric2:
            st.markdown("""
            <div class="metric-card">
                <h3>Genres</h3>
                <h2>{}</h2>
            </div>
            """.format(len(all_genres)), unsafe_allow_html=True)

        with metric3:
            avg_year = int(pd.to_numeric(movies_with_genres['year'], errors='coerce').mean())
            st.markdown("""
            <div class="metric-card">
                <h3>Average Year</h3>
                <h2>{}</h2>
            </div>
            """.format(avg_year), unsafe_allow_html=True)

        # Genre distribution chart
        st.subheader("Genre Distribution")
        genre_fig = plot_genre_distribution(movies_with_genres)
        st.plotly_chart(genre_fig, use_container_width=True)

        # Movie ratings heatmap
        st.subheader("Movie Ratings Heatmap")
        heatmap_fig = plot_rating_heatmap(pivot)
        st.plotly_chart(heatmap_fig, use_container_width=True)

        # User feedback trends
        st.subheader("User Feedback Trends")
        plot_feedback_trends()

    # Tab 3: Movie Explorer
    with tabs[2]:
        st.header("Movie Explorer")

        # Search functionality
        search_term = st.text_input("Search for movies by title, genre, or year")

        if search_term:
            # Filter movies based on search term
            search_results = movies_with_genres[
                movies_with_genres['title'].str.contains(search_term, case=False) |
                movies_with_genres['genres_str'].str.contains(search_term, case=False) |
                movies_with_genres['year'].str.contains(search_term, case=False)
            ]

            if len(search_results) > 0:
                st.write(f"Found {len(search_results)} movies matching '{search_term}'")

                # Display results in a grid
                cols = st.columns(3)
                for i, (_, movie) in enumerate(search_results.iterrows()):
                    with cols[i % 3]:
                        st.markdown(create_movie_card(
                            movie['title'],
                            movie['genres_str'],
                            movie['year']
                        ), unsafe_allow_html=True)

                        # Add a button to get recommendations for this movie
                        if st.button(f"Get Recommendations", key=f"rec_{i}"):
                            st.session_state.selected_movie = movie['title']
                            st.session_state.active_tab = 0
                            st.rerun()
            else:
                st.warning(f"No movies found matching '{search_term}'")

        # Display movies by genre
        st.subheader("Browse by Genre")
        genre_tabs = st.tabs(sorted(list(all_genres))[:5])  # Show top 5 genres

        for i, genre in enumerate(sorted(list(all_genres))[:5]):
            with genre_tabs[i]:
                genre_movies = movies_with_genres[
                    movies_with_genres['genres_list'].apply(lambda x: genre in x)
                ].head(9)  # Show top 9 movies per genre

                genre_cols = st.columns(3)
                for j, (_, movie) in enumerate(genre_movies.iterrows()):
                    with genre_cols[j % 3]:
                        st.markdown(create_movie_card(
                            movie['title'],
                            movie['genres_str'],
                            movie['year']
                        ), unsafe_allow_html=True)

    # Tab 4: A/B Testing
    with tabs[3]:
        st.header("A/B Testing: Traditional vs Advanced Recommendations")

        pivot_trad = get_pivot_traditional()
        movie_list_trad = list(pivot_trad.columns)
        movie_list_trad.sort()

        # Apply filters to A/B testing as well
        if selected_genres or year_range != (min(years), max(years)):
            st.info("Note: Your genre and year filters from the sidebar are applied here too.")

            # Apply the same filtering logic as in the recommendations tab
            if selected_genres:
                filtered_movies = []
                for movie in movie_list_trad:
                    movie_info = movies_with_genres[movies_with_genres['title'] == movie]
                    if not movie_info.empty:
                        movie_genres = movie_info.iloc[0]['genres_list']
                        if any(genre in movie_genres for genre in selected_genres):
                            filtered_movies.append(movie)
                movie_list_trad = filtered_movies

            if year_range:
                filtered_by_year = []
                for movie in movie_list_trad:
                    movie_info = movies_with_genres[movies_with_genres['title'] == movie]
                    if not movie_info.empty:
                        movie_year = movie_info.iloc[0]['year']
                        if movie_year != 'Unknown':
                            if year_range[0] <= int(movie_year) <= year_range[1]:
                                filtered_by_year.append(movie)
                movie_list_trad = filtered_by_year

        if not movie_list_trad:
            st.warning("No movies match your filters. Please adjust your selection.")
        else:
            selected_movie_ab = st.selectbox(
                "Choose a movie for A/B testing", movie_list_trad, key="ab"
            )

            if st.button("Compare Recommendation Methods"):
                try:
                    with st.spinner("Generating recommendations..."):
                        # Get traditional recommendations
                        corr_matrix = compute_similarity(pivot_trad)
                        trad_recs = get_recommendations(
                            selected_movie_ab, pivot_trad, corr_matrix
                        )

                        # Get advanced recommendations
                        adv_recs = advanced_recommendations(
                            selected_movie_ab, pivot_trad
                        )

                        # Prepare dataframes with consistent column names
                        trad_df = trad_recs.reset_index()
                        trad_df = trad_df.rename(columns={
                            selected_movie_ab: "Correlation",
                            'index': 'Movie Title'
                        })

                        adv_df = adv_recs.reset_index()
                        adv_df = adv_df.rename(columns={
                            selected_movie_ab: "Similarity Score",
                            'index': 'Movie Title'
                        })

                        # Create comparison visualization with error handling
                        try:
                            with st.spinner("Creating comparison chart..."):
                                comparison_fig = plot_recommendation_comparison(
                                    selected_movie_ab, trad_recs, adv_recs
                                )
                                st.plotly_chart(comparison_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating comparison chart: {e}")
                            logger.error(f"Error creating comparison chart: {e}")

                        # Show detailed tables with better error handling
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Traditional Recommendations")
                            st.dataframe(trad_df, use_container_width=True)

                        with col2:
                            st.subheader("Advanced Recommendations")
                            st.dataframe(adv_df, use_container_width=True)

                        # Show overlap information with robust error handling
                        try:
                            if 'Movie Title' in trad_df.columns and 'Movie Title' in adv_df.columns:
                                common_movies = set(trad_df['Movie Title']).intersection(set(adv_df['Movie Title']))
                                if common_movies:
                                    st.info(f"The two methods have {len(common_movies)} movies in common.")
                                else:
                                    st.warning("The two recommendation methods returned completely different movies.")
                            else:
                                # Find the actual column names
                                trad_cols = trad_df.columns.tolist()
                                adv_cols = adv_df.columns.tolist()

                                # Try to find the movie title column (it might be 'index' or something else)
                                movie_col_trad = [col for col in trad_cols if col not in ['Correlation', 'Traditional Score']]
                                movie_col_adv = [col for col in adv_cols if col not in ['Similarity Score', 'Advanced Score']]

                                if movie_col_trad and movie_col_adv:
                                    # Use the first column that's not a score column
                                    common_movies = set(trad_df[movie_col_trad[0]]).intersection(set(adv_df[movie_col_adv[0]]))
                                    if common_movies:
                                        st.info(f"The two methods have {len(common_movies)} movies in common.")
                                    else:
                                        st.warning("The two recommendation methods returned completely different movies.")
                                else:
                                    st.warning("Could not determine common movies between methods.")
                        except Exception as e:
                            st.warning(f"Could not compare movie lists: {e}")

                        # Add explanation of methods
                        with st.expander("How do these methods differ?"):
                            st.markdown("""
                            **Traditional Method**: Uses Pearson correlation between movie ratings to find similar movies.
                            This is a direct comparison of rating patterns.

                            **Advanced Method**: Uses Non-negative Matrix Factorization (NMF) to discover latent factors
                            that explain the rating patterns, then computes cosine similarity between movies in this
                            latent space. This can capture more subtle relationships between movies.
                            """)

                except Exception as e:
                    st.error(f"Error in comparison: {e}")
                    logger.error(f"Error in A/B testing: {e}")

    # Footer
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        # Set header color based on theme
        header_color = "#4F6BFF" if st.session_state.get('dark_mode', False) else "#1E3A8A"
        st.markdown(f"<h3 style='color: {header_color};'>About this Dashboard</h3>", unsafe_allow_html=True)
        st.markdown("""
        This enhanced movie recommendation system uses collaborative filtering and matrix factorization
        to provide personalized movie recommendations. The interactive dashboard allows you to explore
        movies, analyze trends, and compare different recommendation algorithms.
        """)

    with col2:
        # Set header color based on theme
        header_color = "#4F6BFF" if st.session_state.get('dark_mode', False) else "#1E3A8A"
        st.markdown(f"<h3 style='color: {header_color};'>Exit Dashboard</h3>", unsafe_allow_html=True)
        if st.button("Exit"):
            st.write("Exiting dashboard...")
            logger.info("Dashboard exit triggered by user.")
            os._exit(0)


if __name__ == "__main__":
    main()
