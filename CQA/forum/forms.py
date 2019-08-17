from django import forms
from .models import Questions


class AddAnswer(forms.ModelForm):
    question = forms.CharField()
    answer = forms.Textarea()

    class Meta:
        model = Questions
        fields = ['question', 'answer']