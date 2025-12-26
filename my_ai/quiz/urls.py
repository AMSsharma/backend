from django.urls import path
from .views import generate_quiz_from_url
from .views import generate_book_chunks_from_url

urlpatterns = [
    path('quiz-generator/', generate_quiz_from_url),
    path('book_chunks/', generate_book_chunks_from_url),
]