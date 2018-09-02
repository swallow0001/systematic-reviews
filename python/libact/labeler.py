"""Interactive Paper Labeler

This module includes an InteractiveLabeler for classifying papers.
"""
from six.moves import input

from libact.base.interfaces import Labeler
from libact.utils import inherit_docstring_from


class InteractivePaperLabeler(Labeler):
    """Interactive Paper Labeler

    InteractivePaperLabeler is a Labeler object that shows the title and
    abstract of a paper and lets human label each feature through command line
    interface.

    Parameters
    ----------
    label_name: list
        Let the label space be from 0 to len(label_name)-1, this list
        corresponds to each label's name.

    """

    def __init__(self, **kwargs):
        self.label_name = kwargs.pop('label_name', None)

    @inherit_docstring_from(Labeler)
    def label(self, feature):

        print(feature)

        banner = "Enter the associated label with the paper: "

        if self.label_name is not None:
            banner += str(self.label_name) + ' '

        lbl = input(banner)

        while (self.label_name is not None) and (lbl not in self.label_name):
            print('Invalid label, please re-enter the associated label.')
            lbl = input(banner)

        return self.label_name.index(lbl)
