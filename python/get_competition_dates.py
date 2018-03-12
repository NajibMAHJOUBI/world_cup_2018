

def get_competition_dates(year):
    if year == 2014:
        return {
            "1st_stage": ["2014/06/12", "2014/06/26"],  # First stage
            "2nd_stage": ["2014/06/28", "2014/07/01"],  # Round of 16
            "3rd_stage": ["2014/07/04", "2014/07/05"],  # Quarter-finals
            "4th_stage": ["2014/07/08", "2014/07/09"],  # Semi-finals
            "5th_stage": ["2014/07/13", "2014/07/13"],  # Final
            "6th_stage": ["2014/07/12", "2014/07/12"]   # Third place
        }
    elif year == 2010:
        return {
            "1st_stage": ["2010/06/11", "2010/06/25"],  # First stage
            "2nd_stage": ["2010/06/26", "2010/06/29"],  # Round of 16
            "3rd_stage": ["2010/07/02", "2010/07/03"],  # Quarter-finals
            "4th_stage": ["2010/07/06", "2010/07/07"],  # Semi-finals
            "5th_stage": ["2010/07/11", "2010/07/11"],  # Final
            "6th_stage": ["2010/07/10", "2010/07/10"]   # Third place
        }
    elif year == 2006:
        return {
            "1st_stage": ["2006/06/09", "2006/06/23"],  # First stage
            "2nd_stage": ["2006/06/24", "2006/06/27"],  # Round of 16
            "3rd_stage": ["2006/06/30", "2006/07/01"],  # Quarter-finals
            "4th_stage": ["2006/07/04", "2006/07/05"],  # Semi-finals
            "5th_stage": ["2006/07/09", "2006/07/09"],  # Final
            "6th_stage": ["2006/07/08", "2006/07/02"]   # Third place
        }
