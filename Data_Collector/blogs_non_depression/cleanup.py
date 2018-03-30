#!/usr/bin/python3

import sys, re

date_re = re.compile(r"(date)? *(?P<day>[0-9]{1,2}),"
                      "(?P<month>February|April|May|June|Juli|July|August|December),"
                      "(?P<year>[0-9]{4})")

month_dict = {"February": "02",
             "April": "04",
             "May": "05",
             "June": "06",
             "Juli": "06",
             "July": "07",
             "August": "08",
             "December": "12",
}

for f in sys.argv[1:]:
    with open(f, 'r') as fp:
        datestr = "00000000"
        posts = {}
        postn = 0
        for line in fp:
            line = line.strip()
            if line:
                m = date_re.match(line)
                if m:
                    olddatestr = datestr
                    datestr = m.group("year") +\
                              month_dict[m.group("month")] + \
                              m.group("day")
                    if olddatestr == datestr:
                        postn += 1
                else:
                    posts[datestr + str(postn)] = posts.get(datestr, " ") + line
        for key in posts:
            if len(posts[key]) > 50: # discard documents less than 50 characters
                fid, rest = f.split(".", maxsplit=1)
                newf = fid + "." + key + "." + rest
                with open(newf, "w") as nfp:
                    print(posts[key], file=nfp)
