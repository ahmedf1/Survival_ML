from django.shortcuts import render
from django.views import View
from django.http import HttpResponse



class homePage(View):
    def get(self,request, *arg, **kwargs):
        context = {}
        return render(request,"template/MainLandingPage.html",context) 

# Create your views here.
