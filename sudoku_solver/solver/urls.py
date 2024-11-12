from django.urls import path
from . import views

urlpatterns = [
    path('api/solve/', views.SudokuSolverView.as_view(), name='solve'),
]
