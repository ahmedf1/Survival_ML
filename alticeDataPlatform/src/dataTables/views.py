from django.shortcuts import render
from django.views import View
from django.http import HttpResponse
from django.views.generic import TemplateView



class homepage(TemplateView):
    template_name = "MainLandingPage.html" 

class dashboardCurrent(TemplateView):
    template_name = "CurrentCustomerDashBoard.html" 

class dashboardNew(TemplateView):
    template_name = "NewAccquiredCustomerDashBoard.html" 

class dashboardProspective(TemplateView):
    template_name = "ProspectiveCustomerDashBoard.html" 

class CustomerProfile(TemplateView):
    template_name = "Customerprofile.html"

# Create your views here.
