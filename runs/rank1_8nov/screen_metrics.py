"""Paired person-equal cell-balanced accuracy on aligned public rows."""
import numpy as np


def compare(data, index, old, new):
    y = data["y"][index]
    people = data["subject"][index]
    streams = data["stream"][index]
    context = data["interface"][index]
    prior, post = old.argmax(1), new.argmax(1)
    result = {}
    for person in np.unique(people):
        cells = []
        for stream in np.unique(streams[people == person]):
            for interface in np.unique(context[streams == stream]):
                cell = (streams == stream) & (context == interface)
                recalls = []
                for label in np.unique(y[cell]):
                    rows = cell & (y == label)
                    recalls.append((np.mean(post[rows] == label), np.mean(prior[rows] == label)))
                cells.append(np.mean(recalls, axis=0))
        current, baseline = 100 * np.mean(cells, axis=0)
        result[person.split("/")[-1]] = {"score": float(current), "delta": float(current - baseline)}
    return result


def blend(base, evidence, weight):
    if weight == 0:
        return base.copy()
    post = base * np.power(evidence, weight)
    return post / post.sum(axis=1, keepdims=True)
