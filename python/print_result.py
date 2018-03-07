
def global_result_by_team(self, data):    
    dic_data_groups = self.get_dic_data_groups()
    groups = sorted(dic_data_groups.keys())
    result_by_team = self.win_losse_drawn_by_team(data)
    dic_result_group_team = {group: {} for group in groups}
    for group in groups:
        print("Group: {0}".format(group))
        for country in dic_data_groups[group]:
            result = team_result(result_by_team[country])
            dic_result_group_team[group][country] = result
            print("Country {0}: {1}".format(country, result))
        print("")
    return dic_result_group_team

def first_second_by_group(self, dic_result_group_team):
    dic_first_by_group, dic_second_by_group = {}, {}
    groups = sorted(dic_result_group_team.keys())
    for group in groups:
        country_result = list(dic_result_group_team[group].iteritems())
        country_result.sort(key=lambda tp: tp[1], reverse=True)
        results = list(np.unique(map(lambda tp: tp[1], country_result)))
        results.sort(reverse=True)
            
        first_teams = filter(lambda tp: tp[1] == results[0], country_result)
        dic_first_by_group[group] = map(lambda tp: tp[0], first_teams)

        if (len(results) >= 2):
            second_teams = filter(lambda tp: tp[1] == results[1], country_result)
            dic_second_by_group[group] = map(lambda tp: tp[0], second_teams)
        else:
            dic_second_by_group[group] = None
    return dic_first_by_group, dic_second_by_group

def print_first_second_by_group(self, dic_first_by_group, dic_second_by_group):
    for group in dic_first_by_group.keys():
        print("Group: {0}".format(group))
        print("1st: {0}".format(dic_first_by_group[group]))
        print("2nd: {0}".format(dic_second_by_group[group]))

def print_matches_next_stage(self, dic_first_by_group, dic_second_by_group):
    def get_team(tp):
        if len(tp) == 1:
            return tp[0]
        else:
            return '-'.join(sentence)

    for tp in self.tp_groups:
        print('/'.join(dic_first_by_group[tp[0]]) + " - " + '/'.join(dic_second_by_group[tp[1]]))
        print('/'.join(dic_first_by_group[tp[1]]) + " - " + '/'.join(dic_second_by_group[tp[0]]))

