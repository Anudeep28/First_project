from django.shortcuts import render, redirect
from .models import Item, appoptions
from .name_model import nameGen
from django.contrib.auth import authenticate, login, logout
# User Creation Form is useful in grabbing all the information from html form tag
from django.contrib.auth.forms import UserCreationForm
# To display messages to the user using Django built in forms
from django.contrib import messages

# initialized the LLM
# call the LLM function
llm = nameGen()

# Create your views here.
def homepage(request):
    # getting all the options to be displayed
    options = appoptions.objects.all()
    # the name assumes you have files under templates folder
    if request.method == 'GET':

        if request.user.is_authenticated:
            option = str(appoptions.objects.get(owner = request.user))
            print(option)
            link_opt = ''.join(i for i in option.split(' '))
            return render(request, template_name='trial1/home.html',
                          context={'option': option, 'link_opt':link_opt, 'options': options})
        else:

            # hence we have started with trial1 instead templates
            return render(request, template_name='trial1/home.html',
                          context={'options': options})

def itemspage(request):
    if request.method == 'GET':
        # getting all the Items created in the admin page to the items variable
        items = Item.objects.filter(owner=None)
        # The context argument passes the variable to the html page
        # it is accessible for use in the html file
        return render(request, template_name='trial1/items.html', \
                      context={'items': items})
    if request.method == 'POST':
        purchased_item = request.POST.get('purchased-item')
        if purchased_item:
            purchased_item_object = Item.objects.get(name=purchased_item)
            purchased_item_object.owner = request.user
            purchased_item_object.save()
            messages.success(request, f'Congratulations you just bought { purchased_item_object.name} for { purchased_item_object.price}$')
        return redirect('items')


def loginpage(request):
    if request.method == 'GET':
        options = appoptions.objects.all()
        return render(request, template_name='trial1/login.html', context = {'options': options})

    if request.method == 'POST':
        # below line gives idea on how i can get the user given info
        # from POST method
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            #messages.success(request, f'You have Successfully logged in as {user.username}')
            # Part where we identify which user has registered for which product
            registered_option = str(appoptions.objects.get(owner = request.user))
            selected_opt = request.POST.get('category')

            if registered_option and registered_option == selected_opt:
                # return the page for the registered option
                messages.success(request, f'You have Successfully logged in as {user.username}')
                return redirect(''.join(i for i in registered_option.split(' ')))
            else:
                messages.error(request, f'You are not Registered for {selected_opt}, Select registered option')
                # redirecting user to items page
                # ** items names should be same as the html filename **
                return redirect('login')
        else:
            messages.error(request, 'Wrong Username or Password')
            return redirect('login')

def registerpage(request):
    if request.method == 'GET':
        options = appoptions.objects.all()
        return render(request, template_name='trial1/register.html', context = {'options': options})
    if request.method == 'POST':
        # grabbing all the information from the forms input tags
        form = UserCreationForm(request.POST)
        # checking if the form is valid for all the required inputs
        if form.is_valid():
            # Save the form to database
            form.save()
            # grabbing the Username
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            option_selected = request.POST.get('category')
            user = authenticate(username=username, password=password)
            login(request, user)

            # Register the user for selected category
            option_confirmed = appoptions.objects.get(name=option_selected)
            option_confirmed.owner = request.user
            option_confirmed.save()

            # messages
            messages.success(request,f'You have Successfully Registered as {user.username}')
            return redirect('login')
        else:
            messages.error(request, form.errors)
            return redirect('register')

def logoutpage(request):
    logout(request)
    messages.success(request, 'You have Successfully Logged out')
    return redirect('home')

def NameGenerator(request):

    if request.method == 'GET':
        name_list = []
        #registered_option = appoptions.objects.get(owner = request.user)

        return render(request, template_name='trial1/NameGenerator.html',context={'name_list':name_list})
    if request.method == 'POST':
        #print(request.POST.get('submit'))
        if request.POST.get('submit') == 'Submit':

            # get the number input by user
            number = int(request.POST.get('number'))
            name_list = llm.gen_name(num=number)

            #print(name_list)
            return render(request, template_name='trial1/NameGenerator.html',context={'name_list':name_list})
        if request.POST.get('clear') == 'Clear':
            #print(request.POST.get('clear'))
            name_list = []
            return render(request, template_name='trial1/NameGenerator.html',context={'name_list':name_list})