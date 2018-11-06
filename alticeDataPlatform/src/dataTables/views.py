from django.shortcuts import render
from django.views import View
from django.http import HttpResponse
from django.views.generic import TemplateView



class homePage(TemplateView):
    template_name = "MainLandingPage.html" 

# Create your views here.
