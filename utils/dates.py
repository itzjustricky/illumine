"""


"""

import datetime
import collections
from numbers import Number

import numpy as np


def split_by(dates, period):
    """ Splits a list of dates periodically

    :param dates: should be a list of dates represented in numerical form of YYYYMMDD
    :param period: decides how often to run the training on the model;
        must take on values ['monthly', 'yearly']
    :return: a dictionary of lists of the dates passed
    """
    if not np.all([isinstance(x, Number) for x in dates]):
        raise ValueError("The passed dates are not in numerical form")
    dates_copy = np.copy(dates)
    allowed_periods = ['monthly', 'yearly']
    if period.lower() not in allowed_periods:
        raise ValueError("Passed an invalid period value, must one of the following {}".format(allowed_periods))
    tomorrow = int((datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y%d%m"))
    if min(dates) <= 19500101 or max(dates) >= tomorrow:
        raise ValueError("There are some invalid dates, min of {} and max of {}"
                         .format(min(dates), max(dates)))

    if period.lower() == 'monthly':
        grp_labels = list(map(dates_by_month, dates_copy))
    if period.lower() == 'yearly':
        grp_labels = list(map(dates_by_year, dates_copy))

    splitted_dates = collections.OrderedDict()
    unique_labels = np.unique(grp_labels)
    for lbl in unique_labels:
        inds = np.where(grp_labels == lbl)[0]
        splitted_dates[lbl] = sorted(dates_copy[inds])

    return splitted_dates


def dates_by_month(date):
    """ Gives the month & year representation of a date
    :param date: should be a date represented in numerical form of YYYYMMDD
    :return: YYYYMM
    """
    return date // 100


def dates_by_year(date):
    """ Gives the year representation of a date
    :param date: should be a date represented in numerical form of YYYYMMDD
    :return: YYYY
    """
    return date // 10000
