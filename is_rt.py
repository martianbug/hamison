# -*- coding: utf-8 -*-
#
# Copyright (C) 2024 Alberto Pérez García-Plaza
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors:
#     Alberto Pérez García-Plaza <alberto.perez@lsi.uned.es>
#
import pandas as pd


def is_rt(row: pd.Series) -> bool:
    """
    Determines if a tweet is a retweet based on the values of the
    user id of the retweeted tweet and the text of the tweet.

    :param row: A row from a DataFrame containing the fields:
        "rt_user_id" and "text"

    :return: True if the tweet is a retweet, False otherwise
    """

    return _is_rt(row['rt_user_id'], row["text"])


def _is_rt(rt_user_id: str, text: str) -> bool:
    """
    Determines if a tweet is a retweet based on the values of the
    user id of the retweeted tweet and the text of the tweet.

    :param rt_user_id:  value coming from the field:
        tweet -> "retweeted_status" -> "user" -> "id"
    :param text: value coming from the field: tweet -> "text"

    :return: True if the tweet is a retweet, False otherwise
    """

    rt: bool = False
    if (rt_user_id != "" and
            not pd.isna(rt_user_id) and
            not pd.isnull(rt_user_id)):

        rt = True

    elif (text.startswith("RT ") or
            text.startswith("#RT ") or
            text.startswith("RT:") or
            " RT " in text):

        rt = True

    return rt