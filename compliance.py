def co2_to_tonnes(kg_co2):
    return kg_co2 / 1000.0

def calculate_credits(tco2):
    # 1 credit per tonne CO2
    return tco2

def check_quota(monthly_emission_kg, quota_kg=50000):
    if monthly_emission_kg <= quota_kg:
        return True, "Within quota"
    else:
        return False, "Quota exceeded"

def compliance_report(monthly_emission_kg):
    approved, reason = check_quota(monthly_emission_kg)
    tco2 = co2_to_tonnes(monthly_emission_kg)
    credits = calculate_credits(tco2)
    return {
        "approved": approved,
        "reason": reason,
        "monthly_emission_kg": monthly_emission_kg,
        "credits_allocated": credits
    }

if __name__ == "__main__":
    # Example
    report = compliance_report(48000)
    print(report)
    report = compliance_report(52000)
    print(report)
