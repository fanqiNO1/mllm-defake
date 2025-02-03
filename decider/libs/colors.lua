local colors = {}

function colors.parse(str, alpha)
	local color = {}
	if type(str) == "string" then
		if str:sub(1, 1) == "#" then
			str = str:sub(2, -1)
		end
		if #str == 8 then
			-- str should be formatted as 'ff00ff80'
			color[1] = tonumber("0x"..string.sub(str,1,2)) / 255
			color[2] = tonumber("0x"..string.sub(str,3,4)) / 255
			color[3] = tonumber("0x"..string.sub(str,5,6)) / 255
			color[4] = tonumber("0x"..string.sub(str,7,8)) / 255
		else
			-- str should be formatted as 'ff00ff'
			color[1] = tonumber("0x"..string.sub(str,1,2)) / 255
			color[2] = tonumber("0x"..string.sub(str,3,4)) / 255
			color[3] = tonumber("0x"..string.sub(str,5,6)) / 255
			color[4] = alpha or 1
		end
	elseif type(str) == "table" then
		for i = 1, #str do
			color[#color + 1] = colors.parse(str[i])
		end
	end
	return color
end

function colors.hsl2rgb(h, s, l, a)
	local r, g, b
	if s == 0 then
		r, g, b = l, l, l -- achromatic
	else
		local function hue2rgb(p, q, t)
			if t < 0 then t = t + 1 end
			if t > 1 then t = t - 1 end
			if t < 1/6 then return p + (q - p) * 6 * t end
			if t < 1/2 then return q end
			if t < 2/3 then return p + (q - p) * (2/3 - t) * 6 end
			return p
		end
		local q
		if l < 0.5 then q = l * (1 + s) else q = l + s - l * s end
		local p = 2 * l - q
		r = hue2rgb(p, q, h + 1/3)
		g = hue2rgb(p, q, h)
		b = hue2rgb(p, q, h - 1/3)
	end
	return {r, g, b, a or 1}
end

function rgb2hsl(r, g, b, a)
	r, g, b = r / 255, g / 255, b / 255
	local max, min = math.max(r, g, b), math.min(r, g, b)
	local h, s, l = (max + min) / 2, (max + min) / 2, (max + min) / 2
	if max == min then
		h, s = 0, 0 -- achromatic
	else
		local d = max - min
		s = l > 0.5 and d / (2 - max - min) or d / (max + min)
		if max == r then
			h = (g - b) / d + (g < b and 6 or 0)
		elseif max == g then
			h = (b - r) / d + 2
		elseif max == b then
			h = (r - g) / d + 4
		end
		h = h / 6
	end
	return {h, s, l, a or 1}
end

function colors.transparent(color, transparency, retainOriginal)
	if color[4] and retainOriginal then
		return color
	else
		return {color[1], color[2], color[3], transparency}
	end
end

function colors.lerp(color1, color2, t)
	if type(color1) == "string" then
		color1 = colors.parse(color1)
	end
	if type(color2) == "string" then
		color2 = colors.parse(color2)
	end
	if type(color1) == "table" and #color1 == 3 then
		color1[4] = 1
	end
	if type(color2) == "table" and #color2 == 3 then
		color2[4] = 1
	end
	local r = color1[1] + (color2[1] - color1[1]) * t
	local g = color1[2] + (color2[2] - color1[2]) * t
	local b = color1[3] + (color2[3] - color1[3]) * t
	local a = color1[4] + (color2[4] - color1[4]) * t
	return {r, g, b, a}
end

function colors.sequenceLerper(colorList, t)
    local numColors = #colorList
    if numColors < 2 then
        error("Need at least two colors for sequence lerping")
    end
    
    -- Normalize t to be within the range of 0 to the number of intervals (numColors - 1)
    t = t % numColors
    local interval = math.floor(t)
    local localT = t - interval

    -- Determine the indices of the colors to lerp between
    local index1 = interval + 1
    local index2 = index1 % numColors + 1

    -- Ensure colors are parsed
	local lColor, rColor
    if type(colorList[index1]) == "string" then
        lColor = colors.parse(colorList[index1])
	else
		lColor = colorList[index1]
    end
    if type(colorList[index2]) == "string" then
		rColor = colors.parse(colorList[index2])
	else
		rColor = colorList[index2]
    end

    -- Perform the lerp between the two selected colors
    return colors.lerp(lColor, rColor, localT)
end

return colors
