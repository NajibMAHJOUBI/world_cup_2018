
def get_features(feature):
    if feature == 'label':
        return "label"
    elif feature == 'features':
        return ["away", "home", "neutral", "losses", "against",
                "for", "draws", "wins"]
