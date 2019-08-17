from django.core.paginator import Paginator
from django.shortcuts import render, redirect
from . import predict
from django.contrib import messages
from .models import Questions
from .forms import AddAnswer


# Create your views here.
def home(request):
    questions = Questions.objects.all().order_by('-id')
    paginate = Paginator(questions, per_page=4)
    page = request.GET.get('page')
    page_obj = paginate.get_page(page)

    context = {
        'qa': questions,
        'page_obj': page_obj
    }
    return render(request, 'forum/home.html', context)


def ask(request):
    global dup
    qpairs = Questions.objects.all()
    questions = list()
    for q in qpairs[::1]:
        questions.append(q.question)
    context = dict()
    context['duplicates'] = None
    context['title'] = 'Home Page'
    print(f"Questions :{questions} ")
    dup = None

    if request.method == 'POST':
        print("entered POST request")
        q = request.POST.get('question')
        predictions, comparision = predict.predict(q, questions)
        for x in comparision:
            if str(x[3]) == '1':
                search = x[0] + 2
                d = x[1]
                dup = Questions.objects.filter(question=d)
                break

        context['predictions'] = predictions
        context['questions'] = questions
        context['original'] = q

        if dup is None:
            context['msg'] = True
            print("Appending the question to the existing list.")
            new_question = Questions(question=q)
            new_question.save()
            # to_append = pd.DataFrame({'test_id': [None], 'question1': [q], 'question2': [None]})
            # data = data.append(to_append, ignore_index=True, sort=False)
            # data.to_csv(r'D:\papers\Project Material\new_try\CQA\forum\test-20.csv', index=False)
        else:
            context['duplicates'] = dup

        print(f"comparisions : {comparision}")
        print("First Question : ", comparision[0][1])
        # print("Type output: "+ type(questions[0][2]))
        print(f"Duplicates {dup}")
        print(type(dup))
        return render(request, 'forum/ask.html', context)

    else:
        return render(request, 'forum/ask.html', context)


def answer(request, qid):
    q = Questions.objects.get(pk=qid)
    print(q.question)
    if request.method == "POST":
        form = AddAnswer(request.POST, instance=q)
        if form.is_valid():
            form.save()
            messages.success(request, f"Your answer has been successfully added")
            return redirect('home')
    else:
        form = AddAnswer(instance=q)

    context = {
        'form': form,
        'qid': qid
    }

    return render(request, 'forum/answer.html', context)



