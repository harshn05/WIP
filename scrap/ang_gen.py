function [] = ang_gen(Ee, pixel, sym)
s = size(Ee);
m = s(1);
n = s(2);

[baseFileName, folder] = uiputfile('*.ang', 'Specify .ang file');
% Create the full file name.
FULL_PATH = fullfile(folder, baseFileName);


% axes(handles.tag_screen);

fid = fopen(FULL_PATH, 'wt');
fprintf(fid, '# MaterialName  	random\n')
fprintf(fid, '# Formula       	random\n')
fprintf(fid, '# Symmetry              %i\n', sym)
fprintf(fid, '# LatticeConstants      1 1 1  90.000  90.000  90.000\n')
fprintf(fid, '# NumberFamilies        0\n')

if(pixel == 'squ')
    progressbar('Writing Ang File');
    for label = 1:m * n
        r = rem(label, m);
        if(r == 0)
            r = m;
        end
        i = r;
        j = 1 + (label - r) / m;
        r = 1;
        x = (i - 1) * r;
        y = (j - 1) * r;
        fprintf(fid, '%.5f\t%.5f\t%.5f\t%d\t%d\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\n', Ee(i, j, 1), Ee(i, j, 2), Ee(i, j, 3), x, y, 255, 1, 1, 0, 0);
        progressbar(label / (m * n));
    end
end

progressbar(1);




if(pixel == 'hex')
    for label = 1:m * n
        r = rem(label, m);
        if(r == 0)
            r = m;
        end
        i = r;
        j = 1 + (label - r) / m;
        R = I(j, i, 1);G = I(j, i, 2);B = I(j, i, 3); % i and j are swapped to make it consistant with TSL
        phi1 = 2 * pi * R;
        phi = pi * G;
        phi2 = 2 * pi * B;
        r = 1;
        x = r * (i - 1) * sqrt(3) + r * (1 - rem(j, 2)) * (sqrt(3)) / 2;
        y = r * (j - 1) * (3 / 2);
        % x = (i - 1) * r;
        % y = (j - 1) * r;
        
        fprintf(fid, '%.5f\t%.5f\t%.5f\t%.4f\t%.4f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\n', phi1, phi, phi2, x, y, 255, 1, 1, 0, 0);
    end
    
end
fclose(fid);
