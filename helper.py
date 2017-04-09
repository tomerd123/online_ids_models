import csv

dsPath=""

with open('E:/thesis_data/datasets/videoJak_full_onlyNetstat.csv', 'rt') as csvin:
    with open('E:/thesis_data/datasets/videoJak_full_onlyNetstat_testSamples.csv', 'w') as csvinWrite:

        #csvin = csv.reader(csvin, delimiter=',')
        for i, row in enumerate(csvin):

            if i%10000==0:
                print(i)
            if (i>1650000 and i<1850000) or i==0:
                csvinWrite.write(row)