clc;
clear;
close all;

lambda = 20; 
noise = 0.001;

% Read in image that shows our environment
img = rgb2gray(imread('room.png'));
% Take all black pixels
[row, col] = find(img==0);
% Scale the image so that each pixel represents a point in space. Pixel
% space defined from 0-100, gpis space defined from -5 to 5 for both x, y
totalObs = [row - 50, col - 50]/5; 

% Convert to polar coordinates
rad = cart2pol(totalObs(:,1), totalObs(:,2));
% Wrap radiance value
radWrapped = mod(rad,2*pi);
% Fix all weird mappings that get floored to 0 and should be 2 pi
radWrapped(radWrapped==0 & rad>0) = 2*pi; 
% Sort the points counterclockwise
[~, sortIdx] = sort(radWrapped, 'ascend');
% rearrange observation array with sorted points
totalObs = totalObs(sortIdx,:);

% Define function for whittle kernel
cov_func = @(pos1, pos2)( pdist2(pos1, pos2)/(2*lambda).*besselk(1, eps+(pdist2(pos1, pos2))*lambda)); 
obs = [];
% Amount of frames the points should be constructed in
steps = 50;
% Loop until all points are used as observation
for i=1:steps
    % points per step
    interv = int32(floor(size(totalObs,1)/steps));
    
    % Take in all observations including the ones from current step
    obs = totalObs(1:1:i*interv,:); 
    N_obs = size(obs, 1); 

    % Testing points
    [X, Y] = meshgrid(-10:0.1:10, -10:0.1:10);
    Qpoint(:,1) = X(:);
    Qpoint(:,2) = Y(:);
    
    % Calculate covariances
    K = cov_func(obs, obs); 
    k = cov_func(Qpoint, obs); 

    % gp regression 
    fprintf('Start gaussian regression!\n');
    tic
    y = 1 + noise*randn(size(obs, 1), 1);
    mu = k * ((K + noise * eye(N_obs)) \ y); 

    % calculate mean and apply log
    mean_dist = -(1 / lambda) * log(abs(mu));

    toc

    fprintf('Finish gaussian regression!\n\n');

    hold on;
    cla;
    % colormap for surf
    colormap(hsv);
    colorbar;
    alpha 0.8;
    surf(X, Y, reshape(mean_dist, size(X)), 'EdgeColor', 'None' ); 
    
    % Plot observation points
    alpha 1;
    % Old observations are black
    if(i>1)
        plot(obs(1:((i-1)*interv)+1, 1), obs(1:((i-1)*interv)+1, 2), 'ko', 'MarkerFaceColor', 'k'); 
    end
    % New observations are red
    plot(obs(((i-1)*interv)+1:i*interv, 1), obs(((i-1)*interv)+1:i*interv, 2), 'ko', 'MarkerFaceColor', 'r'); 
    % Fix limits and view
    xlim([-10, 10])
    ylim([-10, 10])
    zlim([0, 15])
    view(-170,50);
    grid on;
    
    % Set shading settings for beautiful plotting
    shading interp;
    camlight; 
    lighting phong;

    set(gca,'FontSize',15);
    title('DistanceField from loaded Image');
    
    % create the gif 
    % needs https://de.mathworks.com/matlabcentral/fileexchange/63239-gif
    if(i == 1)
        gif('animatedDistField.gif', 'DelayTime',1/10)
    else
        gif
    end
end