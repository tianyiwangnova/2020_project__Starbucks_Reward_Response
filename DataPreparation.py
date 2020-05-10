class DataPreparation:
    
    def __init__(self):
        pass
    
    @staticmethod
    def bucket_age(x):
        x = float(x)
        if x < 25 :
            return "0 ~ 24"
        elif x >= 25 and x <= 34:
            return "25 ~ 29"
        elif x >= 35 and x <= 44:
            return "35 ~ 44"
        elif x >= 45 and x <= 54:
            return "45 ~ 54"
        elif x >=55 and  x <= 64:
            return "55 ~ 64"
        elif x is None:
            return float('nan')
        else:
            return "65 ~"
    
    def fit(self, data):
        self.data = data
        self.most_common_age = data['age'].mode()[0]
        self.data[['offers_viewed_before','offers_completed_before']] =  self.data[['offers_viewed_before','offers_completed_before']].fillna(0)
        self.data[['hours_since_last_viewed','hours_since_last_completed']] = self.data[['hours_since_last_viewed','hours_since_last_completed']].fillna(-1)
        self.data['age'] = self.data['age'].fillna(self.most_common_age)
        self.data['age_bucket'] = self.data['age'].apply(self.bucket_age)
        self.median_income_age_buckets = self.data.groupby('age_bucket')['income'].median().reset_index()
        self.data.loc[self.data['income'].isna(), 'income'] = self.data[self.data['income'].isna()]['age_bucket']\
                                                                         .apply(lambda x: self.median_income_age_buckets.loc[self.median_income_age_buckets['age_bucket'] == x, "income"].values[0])
    def fit_transform(self, data):
        self.fit(data)
        return self.data.drop('age_bucket', axis=1)
    
    def transform(self, new_data):
        new_data[['offers_viewed_before','offers_completed_before']] =  new_data[['offers_viewed_before','offers_completed_before']].fillna(0)
        new_data[['hours_since_last_viewed','hours_since_last_completed']] = new_data[['hours_since_last_viewed','hours_since_last_completed']].fillna(-1)
        new_data['age'] = new_data['age'].fillna(self.most_common_age)
        new_data['age_bucket'] = new_data['age'].apply(self.bucket_age)
        new_data.loc[new_data['income'].isna(), 'income'] = new_data[new_data['income'].isna()]['age_bucket']\
                                                                         .apply(lambda x: self.median_income_age_buckets.loc[self.median_income_age_buckets['age_bucket'] == x, "income"].values[0])
        return new_data.drop('age_bucket', axis=1)