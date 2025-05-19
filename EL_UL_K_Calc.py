from scipy.stats import norm

def EL_Calc(calibrated_pd):
    # Voeg een kolom toe voor de geconfigureerde LGD en EAD
    # Voeg LGD en EAD kolommen toe aan de DataFrame
    calibrated_pd['LGD'] = 1  # Voorbeeldwaarde, vervang met je eigen LGD waarden; 1 zodat deze niet meetelt in de berekening
    calibrated_pd['EAD'] = 1  # Voorbeeldwaarde, vervang met je eigen EAD waarden; 1 zodat deze niet meetelt in de berekening
    # Bereken het verwachte verlies
    calibrated_pd['EL'] = calibrated_pd['method Cal_PD'] * calibrated_pd['LGD'] * calibrated_pd['EAD']
    # Print de eerste paar rijen van de DataFrame met de verwachte verlies
    #print(calibrated_pd.head())
    # Bereken het totale verwachte verlies
    total_expected_loss = calibrated_pd['EL'].sum()
    #print(f'Total Expected Loss: {total_expected_loss}')
    return calibrated_pd, total_expected_loss

def UL_Calc(calibrated_pd):
    R = 0.15 #voor retail exposures
    # Voeg een kolom toe voor de geconfigureerde LGD en EAD
    # Voeg LGD en EAD kolommen toe aan de DataFrame
    calibrated_pd['LGD'] = 1  # Voorbeeldwaarde, vervang met je eigen LGD waarden; 1 zodat deze niet meetelt in de berekening
    calibrated_pd['EAD'] = 1  # Voorbeeldwaarde, vervang met je eigen EAD waarden; 1 zodat deze niet meetelt in de berekening
    # Bereken het onverwachte verlies
    # De IRB-formule herschreven met juiste distributiefuncties:
    calibrated_pd['UL'] = (
        calibrated_pd['LGD'] *
        (norm.cdf(
            (1 - R)**(-0.5) * norm.ppf(calibrated_pd['method Cal_PD']) +
            (R / (1 - R))**0.5 * norm.ppf(0.999)) - calibrated_pd['method Cal_PD'])*calibrated_pd['EAD'] # op het einde * EAD want K is een percentage
    ) 
    # Print de eerste paar rijen van de DataFrame met de onverwachte verlies
    #print(calibrated_pd.head())
    # Bereken het totale onverwachte verlies
    total_UL = calibrated_pd['UL'].sum()
    #print(f'Total UL: {total_UL}')
    return calibrated_pd, total_UL