
def get_matches(year):
    if year == "2018":
        matches = {"1st_stage": {}}
        matches["1st_stage"]["A"] = [("Russia", "Saudi Arabia"),
                                     ("Egypt", "Uruguay"),
                                     ("Russia", "Egypt"),
                                     ("Uruguay", "Saudi Arabia"),
                                     ("Uruguay", "Russia"),
                                     ("Saudi Arabia", "Egypt")]
        matches["1st_stage"]["B"] = [("Morocco", "Iran"),
                                     ("Portugal", "Spain"),
                                     ("Portugal", "Morocco"),
                                     ("Iran", "Spain"),
                                     ("Iran", "Portugal"),
                                     ("Spain", "Morocco")]
        matches["1st_stage"]["C"] = [("France", "Australia"),
                                     ("Peru", "Denmark"),
                                     ("Denmark", "Australia"),
                                     ("France", "Peru"),
                                     ("Denmark", "France"),
                                     ("Australia", "Peru")]
        matches["1st_stage"]["D"] = [("Argentina", "Iceland"),
                                     ("Croatia", "Nigeria"),
                                     ("Argentina", "Croatia"),
                                     ("Nigeria", "Iceland"),
                                     ("Nigeria", "Argentina"),
                                     ("Iceland", "Croatia")]
        matches["1st_stage"]["E"] = [("Costa Rica", "Serbia"),
                                     ("Brazil", "Switzerland"),
                                     ("Brazil", "Costa Rica"),
                                     ("Serbia", "Switzerland"),
                                     ("Serbia", "Brazil"),
                                     ("Switzerland", "Costa Rica")]
        matches["1st_stage"]["F"] = [("Germany", "Mexico"),
                                     ("Sweden", "South Korea"),
                                     ("Germany", "Sweden"),
                                     ("South Korea", "Mexico"),
                                     ("South Korea", "Germany"),
                                     ("Mexico", "Sweden")]
        matches["1st_stage"]["G"] = [("Belgium", "Panama"),
                                     ("Tunisia", "England"),
                                     ("Belgium", "Tunisia"),
                                     ("England", "Panama"),
                                     ("England", "Belgium"),
                                     ("Panama", "Tunisia")]
        matches["1st_stage"]["H"] = [("Poland", "Senegal"),
                                     ("Colombia", "Japan"),
                                     ("Poland", "Colombia"),
                                     ("Japan", "Senegal"),
                                     ("Japan", "Poland"),
                                     ("Senegal", "Colombia")]
        return matches

