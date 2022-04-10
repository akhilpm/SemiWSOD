from visdom import Visdom
import numpy as np

class VisdomLinePlotter(object):
    """Plots to Visdom
       var_name: variable name (e.g. loss, acc)
       split_name: split name (e.g. train, val)
       title_name: titles of the graph (e.g. Classification Accuracy)
       x: x axis value (e.g. epoch number)
       y: y axis value (e.g. epoch loss) """
    def __init__(self, env_name='main', port=8097):
        self.viz = Visdom(port=port)
        self.viz.close(None)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y, line_name):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, name=split_name+"_"+line_name, opts=dict(
                legend=[split_name+"_"+line_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name+"_"+line_name, update='append')


class VisdomHistogramPlotter(object):
    def __init__(self, env_name="main", port=8097, win="histogram"):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.win = win

    def plot(self, X, numbins):
        self.viz.histogram(X, env=self.env, win=self.win, opts=dict(numbins=numbins))


class VisdomBarplotter(object):
    def __init__(self, env_name="main", port=8097, win="bar"):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.win = win
        self.legend = ["total", "corloc STN", "corloc NT"]
        #self.legend = ["total", "corloc STN"]

    def plot(self, x, y):
        self.viz.bar(X=x, Y=y, env=self.env, win=self.win, opts=dict(stacked=False, legend=self.legend))


class VisdomImagePlotter(object):
    def __init__(self, env_name="main", port=8097, win="images"):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.win=win

    def plot(self, images, caption, nrows=3):
        self.viz.images(images, nrow=nrows, padding=3, env=self.env, win=self.win, opts=dict(caption=caption))