from __future__ import division
import csv
import os
import pandas as pd
from unicodedata import normalize
from collections import Counter
from operator import itemgetter, attrgetter, methodcaller
from langdetect import detect  
import unidecode
import json
import sys
import requests
import time

class TwitterData_Initialize:
    data = []
    processed_data = []
    wordlist = []

    data_model = None
    data_labels = None
    is_testing = False

    mapsKey = None
   
    def remove_bad_rows(self,csv_file, is_testing_set=False):
        
        if not is_testing_set:
            # We will eliminate the rows that are more than 11 elements

            input_file = open(csv_file, 'r')
            reader = csv.reader(input_file, delimiter=';')

            temp_file= open("data/temp.txt", 'w') 
            writer = csv.writer(temp_file, delimiter=";")
            row1 = next(reader)
            n_rows = len(row1)
            writer.writerow(row1)

            print("Removing bad rows. Numer of headers = "+ str(n_rows))

            count = 0
            for row in reader:
                if len(row) != n_rows: 
                    count += 1
                else:
                    writer.writerow(row)

            print("Cleaned, eliminated: " + str(count) + " rows")
            
            input_file.close()
            temp_file.close()
            os.remove(csv_file)
            os.rename("data/temp.txt", csv_file)


            df = pd.read_csv(csv_file,sep =';')
            df.drop_duplicates('id', keep = 'first', inplace = True)
            df.to_csv(csv_file, sep = ";", index = False)
    
    def add_coordinates_col(self,csv_file, is_testing_set=False):
        if not is_testing_set:
            input_file = open(csv_file) 
            reader = csv.reader(input_file, delimiter=';')
            row1 = next(reader) 
            if not 'coordinates' in row1:
                df = pd.read_csv(csv_file, sep = ";")
                
                locations = df['location']
                locations.fillna('', inplace = True)
                locations = pd.unique(df[['location']].values.ravel('K'))
                print("Debug: " + str(len(locations)) + " locations to process")
                #print("DEBUG: locations array is:")
                #print(str(locations))
                coordinates = []
                mydict = {}
                mydict2 = {}
                dictLocation = {}
                i = 0

               
                with open("data/dictLocation.txt", "r") as csv_file1:
                    reader = csv.reader(csv_file1, delimiter=',')
                    for row in reader:
                        print("ROW ---->"+str(row[0]))
                        values = row[0].split(";")
                        print('values---------------------->'+str(values))
                        use = values[0].split(',')
                        if(len(use)>=2):
                            dictLocation[str(use[0]+','+use[1])] = str(values[1])


                with open("data/mydict.txt", "r") as csv_file2:
                    reader = csv.reader(csv_file2, delimiter=',')
                    for row in reader:
                        #print("ROW ---->"+str(row[0]))
                        values = row[0].split(";")
                        #print('values USE---------------------->'+str(values[0]))
                        use = values[1].split(',')
                        if(len(use)>=2):
                            mydict[str(values[0])] = str(use[0]+','+use[1])
                            mydict2[str(use[0]+', '+use[1])] = values[1]

                reader = csv.reader(open('data/dictLocation.txt','r'), delimiter=',')
                data = list(reader)
                row_count = len(data)
                locations = locations[(row_count)+1:]
                print("ROW COUNT -----> "+str(row_count))

                
                writer=csv.writer(open('data/dictLocation.txt','a'), delimiter=',')
                writer2=csv.writer(open('data/mydict.txt','a'), delimiter=',')
                

                for location in locations:
                    i += 1
                    if location != "":
                        if(i%20 == 0):
                            print("At entry " +str(i)+" of total  of file...")
                        coordinate = self.getLatLong(location)
                        # Trasform the place name to any langauge to english, thanks to maps api
                        if coordinate != "NOTFOUND":
                            latlog = coordinate.split(",")
                            calculatedLocation = self.getplace(latlog[0],latlog[1])
                        else:
                            calculatedLocation = "NOTFOUND"
                        
                       
                        dictLocation[coordinate] = calculatedLocation
                        writer.writerow([coordinate+";"+calculatedLocation])

                        # -------------
                       
                        mydict[location] = coordinate
                        writer2.writerow([location+";"+coordinate])

                print('my dict 2 ---->'+str(mydict2))

                df['coordinates'] = df['location'].astype(str).map(mydict)
                df['location'] = df['coordinates'].astype(str).map(dictLocation)

                df.to_csv(r"data/output.txt",header = True, sep = ";",index = None)

                os.remove(csv_file)
                os.rename("data/output.txt", csv_file)
            else:
                print("Coordinates already in train, skipped")
                input_file.close()
            
    def getKeyFromConfigFile(self,keyName,num):
        path = "./config/config.txt"
        if os.path.exists(path):
            print("Importing key from config file")
            configFile = open(path,'r')
            next(configFile,None)
            count = 0
            for row in configFile:
                temp = row.split(":")
                key = temp[0]
                value = temp[1]
                if key == keyName:
                    if value != "" and value is not None:
                        if count == num:
                            print("Key found:\n" + row)
                            return value
                        else:
                            count += 1
                else:
                    print(keyName + " not found, add row with that key in: " + path)
                    sys.exit(0)
        else:
            print(path + " not found. Add the file in order to execute the program")
            sys.exit(0)

    def getLatLong(self,city):
        city = city.replace(" ", "+") # delete space
        headers = {'user-agent': 'test'}
        if self.whichKey == 0:
            keyToUse = self.mapsKey0
        elif self.whichKey == 1:
            keyToUse = self.mapsKey1
        elif self.whichKey == 2:
            keyToUse = self.mapsKey2
        else:
            print("Error, whichKey not coerent")
            keyToUse = self.mapsKey0
        

        url = "https://maps.googleapis.com/maps/api/geocode/json?"
        url += "address=%s&sensor=false&key=%s" % (city, keyToUse[:-1])
        #url += "address=%s&sensor=false" % (city)
        response = requests.get(url, headers=headers)
        jsonResponse = json.loads(response.text)
        status = jsonResponse['status']
        while status != "OK": #    "status": "OVER_QUERY_LIMIT"
            print(status)
            if status == "ZERO_RESULTS":
                print("Error at city: %s" % (city))
                return "NOTFOUND"
            elif status == "OVER_QUERY_LIMIT":

                self.whichKey = (self.whichKey + 1) % 3
                if self.whichKey == 0:
                    keyToUse = self.mapsKey0
                elif self.whichKey == 1:
                    keyToUse = self.mapsKey1
                elif self.whichKey == 2:
                    keyToUse = self.mapsKey2
                else:
                    print("Error, whichKey not coerent")
                    keyToUse = self.mapsKey0

                print("Query limit reached, change the key with the %s" % self.whichKey)
                url = "https://maps.googleapis.com/maps/api/geocode/json?"
                url += "address=%s&sensor=false&key=%s" % (city, keyToUse[:-1])
                response = requests.get(url, headers=headers)
                jsonResponse = json.loads(response.text)
                status = jsonResponse['status']
            else:
                '''
                print("Status is: " + status)
                print("Key limit, waiting..")
                time.sleep(1)
                response = requests.get(url, headers=headers)
                jsonResponse = json.loads(response.text)
                status = jsonResponse['status']
                '''
                return "NOTFOUND"
        components = jsonResponse['results'][0]['geometry']['location']
        lat = lng = None
        lat = components['lat']
        lng = components['lng']
        return "%s,%s" % (lat, lng)
    
    def getplace(self,lat, lon):
        lat = lat.replace(" ", "+")
        lon = lon.replace(" ", "+")
        headers = {'user-agent': 'test'} 

        if self.whichKey == 0:
            keyToUse = self.mapsKey0
        elif self.whichKey == 1:
            keyToUse = self.mapsKey1
        elif self.whichKey == 2:
            keyToUse = self.mapsKey2
        else:
            print("Error, whichKey not coerent")
            keyToUse = self.mapsKey0

        url = "https://maps.googleapis.com/maps/api/geocode/json?"
        url += "latlng=%s,%s&sensor=false&key=%s" % (lat, lon,keyToUse[:-1])
        response = requests.get(url, headers=headers)
        jsonResponse = json.loads(response.text)

        status = jsonResponse['status']
        while status != "OK": #    "status": "OVER_QUERY_LIMIT"
            if status == "ZERO_RESULTS":
                #print("Error at city: %s" % (city))
                return "NOTFOUND"
            elif status == "OVER_QUERY_LIMIT":
                self.whichKey = (self.whichKey + 1) % 3
                if self.whichKey == 0:
                    keyToUse = self.mapsKey0
                elif self.whichKey == 1:
                    keyToUse = self.mapsKey1
                elif self.whichKey == 2:
                    keyToUse = self.mapsKey2
                else:
                    print("Error, whichKey not coerent")
                    keyToUse = self.mapsKey0
                print("Query limit reached, change the key with the %s" % keyToUse[:-1])
                url = "https://maps.googleapis.com/maps/api/geocode/json?"
                url += "address=%s&sensor=false&key=%s" % (city, keyToUse[:-1])
                response = requests.get(url, headers=headers)
                jsonResponse = json.loads(response.text)
                status = jsonResponse['status']
            else:
                '''
                print("Status is: " + status)
                print("Key limit, waiting..")
                time.sleep(1)
                response = requests.get(url, headers=headers)
                jsonResponse = json.loads(response.text)
                status = jsonResponse['status']
                '''
                return "NOTFOUND"
        components = jsonResponse['results'][0]['address_components']
        country = town = region = None
        for c in components:
            if "country" in c['types']:
                country = c['long_name']
            if "locality" in c['types']:
                town = c['long_name']
            if "administrative_area_level_1" in c['types']:
                region = c['short_name']

        return "%s,%s,%s" % (town, country,region)
                
    def add_label_col(self,csv_file, is_testing_set=False, is_spain = False):
        if not is_testing_set:
            #"votarem"
            positive  = ["votarem","empaperem","independencia","freepiolin",
            "viscacatalunya","sensepor","volemvotar","votaremiguanyarem"]

            negative = ["prenpartit","espanasalealacalle","hispanofobia", 
            "madcataluña","catalunyaesespana","piquefueradelaseleccion",
            "espanaunida","hispanofobia","culturayciudadania",
            "espananoserompe","yosiquieroserespanol","espanolesorgullosos"] 
            
            catalan_provinces = ['barcelona','gerona','lérida','lerida','tarragona']

            input_file = open(csv_file) 
            reader= csv.reader(input_file, delimiter=';')
            row1 = next(reader)

            there_is_sentiment = False
            for header in row1:
                if header == 'sentiment':
                    there_is_sentiment = True

            if not there_is_sentiment:
                temp_file= open('data/temp.txt', 'w') 
                writer = csv.writer(temp_file, delimiter=";")
                row1.append('sentiment')
                writer.writerow(row1)

                index_text = row1.index('text')
                index_language = row1.index('language')
                index_location = row1.index('location')
                index_language = row1.index('language')
                print("Adding sentiment entries. Text row is: "+ str(index_text))

                count_new = 0
                for row in reader:
                    condition = False
                    count_negative = 0.0
                    count_positive = 0.0

                    if not is_spain:
                        condition = 'ca' in row[index_language]
                    else:
                        condition = 'es' in row[index_language]
                   
                    if(condition == True):
                        count_new +=1
                        #print(str(row[index_location]))
                        for i in range(0,len(positive)):
                            count_positive+= unidecode.unidecode(row[index_text].lower()).count("#"+positive[i])
                            if(unidecode.unidecode(row[index_text].lower()).count("votarem")>2):
                                count_positive-=1
                        for i in range(0,len(negative)):
                            count_negative+= unidecode.unidecode(row[index_text].lower()).count("#"+negative[i])
                        
                        if(count_negative>0 or count_positive>0):
                            percentage_negative = float(count_negative)/(count_negative+count_positive)
                            percentage_positive = float(count_positive)/(count_negative+count_positive)
                            if(percentage_positive>0.5):
                                new_row = row
                                new_row.append('positive')
                                writer.writerow(new_row)
                            elif(percentage_negative> 0.5):
                                new_row = row
                                new_row.append('negative')
                                writer.writerow(new_row)
                            else:
                                new_row = row
                                new_row.append('neutral')
                                writer.writerow(new_row)

                print("Finished adding sentiment entries. Number values = "+str(count_new))  

                input_file.close()
                temp_file.close()
                os.remove(csv_file)
                os.rename("data/temp.txt", csv_file)
            else:
                input_file.close()
    
    def initialize(self, csv_file, is_testing_set=False, from_cached=None, is_spain = False):
        #self.mapsKey = self.getKeyFromConfigFile("mapsKey")
        if not is_testing_set:
            #self.remove_bad_rows(csv_file)
            #self.add_language_col(csv_file)
            #self.add_coordinates_col(csv_file)
            self.add_label_col(csv_file, is_spain=is_spain)

        if from_cached is not None:
            self.data_model = pd.read_csv(from_cached,sep = ';')
            return

        if not os.path.exists('plot'):
            os.makedirs('plot')

        self.is_testing = is_testing_set
		

        if not is_testing_set:
            self.data = pd.read_csv(csv_file, sep = ';',header=0)
            self.data = self.data[self.data["sentiment"].isin(['positive','negative','neutral'])]
        else:
            self.data = pd.read_csv(csv_file, header=0, names=["id", "text"],dtype={"id":"int64","text":"str"},nrows=4000)
            not_null_text = 1 ^ pd.isnull(self.data["text"])
            not_null_id = 1 ^ pd.isnull(self.data["id"])
            self.data = self.data.loc[not_null_id & not_null_text, :]

        self.processed_data = self.data
        self.wordlist = [] 
        self.data_model = None
        self.data_labels = None