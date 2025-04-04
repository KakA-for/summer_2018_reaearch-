function [result] = scale(input)
 result = 2*(input-mean(input))./(max(input)-min(input));
end

