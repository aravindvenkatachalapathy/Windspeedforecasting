import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculation_s_rr(f, windfield, parameter):
    """
    Calculates the rotor effective wind speed spectrum for a given Kaimal wind field and turbine.
    
    Parameters:
        f (numpy array): Frequency vector [Hz]
        windfield (dict): Dictionary containing the grid of the wind field
        parameter (dict): Dictionary containing the turbine parameters
        
    Returns:
        numpy array: Rotor effective wind speed spectrum
    """
    URef = parameter['TurbSim']['URef']
    IRef = parameter['TurbSim']['IRef']
    R = parameter['Turbine']['R']
    Y = windfield['grid']['Y']
    Z = windfield['grid']['Z']
    
    # Kaimal spectrum parameters
    Lambda_1 = 42
    L_1 = 8.1 * Lambda_1
    b = 5.6
    sigma_1 = IRef * (0.75 * URef + b)
    
    L_c = L_1
    a = 12
    kappa = a * np.sqrt((f / URef)**2 + (0.12 / L_c)**2)
    S_uu = 4 * L_1 / URef / ((1 + 6 * f * L_1 / URef)**(5/3)) * sigma_1**2
    
    # Points in rotor disc
    distance_from_hub = np.sqrt(Z.flatten()**2 + Y.flatten()**2)
    points_in_rotor_disc = distance_from_hub <= R
    n_points = len(distance_from_hub)
    n_points_in_rotor_disc = np.sum(points_in_rotor_disc)
    
    # Loop over all points
    sum_gamma = np.zeros_like(f)
    for i in range(n_points):
        if points_in_rotor_disc[i]:
            for j in range(n_points):
                if points_in_rotor_disc[j]:
                    distance = np.sqrt((Y.flatten()[j] - Y.flatten()[i])**2 + (Z.flatten()[j] - Z.flatten()[i])**2)
                    sum_gamma += np.exp(-kappa * distance)
    
    # Final spectrum
    return sum_gamma * S_uu / n_points_in_rotor_disc**2

def generate_rotor_effective_wind_speed(windfield, parameter):
    """
    Generates a time series for rotor-effective wind speed.
    
    Parameters:
        windfield (dict): Dictionary containing the grid of the wind field
        parameter (dict): Dictionary containing the simulation and turbine parameters
    
    Returns:
        dict: Contains time series data of rotor-effective wind speed
    """
    T = parameter['Time']['TMax']
    dt = parameter['Time']['dt']
    seed = parameter['TurbSim']['RandSeed']
    URef = parameter['TurbSim']['URef']
    
    # Frequency setup
    f_min = 1 / T
    f_max = 1 / (2 * dt)
    df = f_min
    f = np.arange(f_min, f_max + df, df)
    
    # Spectrum
    S_RR = calculation_s_rr(f, windfield, parameter)
    
    # Amplitudes
    A = np.sqrt(2 * S_RR * df)
    
    # Random phase angles
    np.random.seed(seed)
    Phi = np.random.rand(len(f)) * 2 * np.pi
    
    # Time vector
    t = np.arange(0, T, dt)
    
    # Generate time series
    U = len(f) * np.concatenate(([0], A)) * np.exp(np.concatenate(([0], 1j * Phi)))
    v_0 = URef + np.fft.ifft(U, len(t)).real
    
    # Prepare output
    return {'time': t, 'v_0': v_0}

def main():
    # Configuration
    parameter = {
        'Turbine': {'R': 63},
        'Time': {'dt': 0.1, 'TMax': 3600},
        'TurbSim': {'IRef': 0.16}
    }
    
    # Windfield grid
    y, z = np.meshgrid(np.arange(-64, 65, 8), np.arange(-64, 65, 8))
    windfield = {'grid': {'Y': y, 'Z': z}}
    
    # Mean wind speeds for DLC 1.2
    URef_v = np.arange(4, 26, 2)
    
    plt.figure()
    plt.xlabel('t [s]')
    plt.ylabel('v_0 [m/s]')
    
    # Loop over wind speeds
    for URef in URef_v:
        # Update parameters
        parameter['TurbSim']['URef'] = URef
        parameter['TurbSim']['RandSeed'] = URef
        
        print(f'Generating {URef} m/s!')
        
        # Generate rotor-effective wind speed
        disturbance = generate_rotor_effective_wind_speed(windfield, parameter)
        
        # Plot the disturbance
        plt.plot(disturbance['time'], disturbance['v_0'])
        plt.draw()
        
        # Save disturbance to CSV
        df = pd.DataFrame({'time': disturbance['time'], 'v_0': disturbance['v_0']})
        df.to_csv(f'URef_{URef:02d}_Disturbance.csv', index=False)
    
    plt.show()

if __name__ == "__main__":
    main()
