from django.urls import path
from .views import PerguntaView

urlpatterns = [
    path('ask/', PerguntaView.as_view()),
]
