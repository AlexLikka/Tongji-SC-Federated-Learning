function H = UE_Channel(Nr, Nt, Npaths, angle_spread, space)
    % Nt: Number of transmit antennas
    % Nr: Number of receive antennas
    % Npaths: Number of paths in the channel
    % angle_spread: Spread of angles in degrees
    % d: Antenna spacing (in wavelengths)

    % Initialize channel matrix
    H = zeros(Nr, Nt);
    
    % Define antenna array vectors
    tx_array = (0:Nt-1).' * space;
    rx_array = (0:Nr-1).' * space;

    % Loop over the number of paths
    for p = 1:Npaths
        % Generate random angles of departure and arrival
        angle_of_departure = -angle_spread/2 + angle_spread*rand();
        angle_of_arrival = -angle_spread/2 + angle_spread*rand();

        % Generate random path gain
        path_gain = (randn() + 1j*randn())/sqrt(2);

        % Calculate array response vectors
        at = exp(-1j * 2 * pi * tx_array * sin(deg2rad(angle_of_departure)));
        ar = exp(-1j * 2 * pi * rx_array * sin(deg2rad(angle_of_arrival)));

        % Add the contribution of this path to the channel matrix
        H = H + path_gain * ar * at';
    end
    % normalization
    H = H * sqrt(Nt*Nr/Npaths);