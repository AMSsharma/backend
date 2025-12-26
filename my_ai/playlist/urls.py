from django.urls import path
from .views import ExtractPlaylistView
from .views import CalculateTaskDifficultyView
urlpatterns = [
    path("extract-playlist/", ExtractPlaylistView.as_view(), name="extract_playlist"),
    path("calculate-difficulty/", CalculateTaskDifficultyView.as_view(), name="generate_score"),
]