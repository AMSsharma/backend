from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('quiz.urls')), 
    path('playlist/', include('playlist.urls')), # 👈 Route to your app
]
