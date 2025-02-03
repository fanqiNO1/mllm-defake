DEBUG_ANDROID = false
local submit = require("submit")
local json = require("libs.json")
local lg = love.graphics
local font = lg.newFont("QuickSand.ttf", lg.getHeight() * 0.1)
local fontSmall = lg.newFont("QuickSand.ttf", 14)
local easing = require("libs.easing")

local fakeImages = love.filesystem.getDirectoryItems("images/fake")
local realImages = love.filesystem.getDirectoryItems("images/real")
table.sort(fakeImages)
table.sort(realImages)
local allImages = {}
for i = 1, #fakeImages do
    allImages[#allImages + 1] = "images/fake/"..fakeImages[i]
end
for i = 1, #realImages do
    allImages[#allImages + 1] = "images/real/"..realImages[i]
end
local rng = love.math.newRandomGenerator(114)
local function getRandomInt(min, max)
    return rng:random(min, max)
end
local function shuffleList(list)
    for i = 1, #list do
        local j = getRandomInt(1, #list)
        list[i], list[j] = list[j], list[i]
    end
end
shuffleList(allImages)

local currentImage = 1
local images = {}
local state = {}
local stats = {
    rr = 0,
    rf = 0,
    fr = 0,
    ff = 0
}
local accs = {}
local touchDelta = {
    x = 0,
    y = 0,
    lastReleaseT = 0,
    action = nil,
    lockId = -1
}
local selectOnRelease = {
    left = false,
    right = false
}




local function getGT(imagePath)
    local isReal = imagePath:find("real")
    if isReal then
        return 1
    else
        return 0
    end
end

local function loadState()
    local data = love.filesystem.read("decider.json")
    if data then
        state = json.decode(data)
        return
    end
    state = {
        pointer = 1,
        decisions = {}
    }
end

local function saveState()
    love.filesystem.write("decider.json", json.encode(state))
end
loadState()

if state.finished then
    love.window.setTitle("Decider | Finished")
end

local function computeStatsFromState(state)
    local stats = {
        rr = 0,
        rf = 0,
        fr = 0,
        ff = 0
    }
    for k, v in pairs(state.decisions) do
        local gt = getGT(k)
        if gt == 1 and v == 1 then
            stats.rr = stats.rr + 1
        elseif gt == 1 and v == 0 then
            stats.rf = stats.rf + 1
        elseif gt == 0 and v == 1 then
            stats.fr = stats.fr + 1
        elseif gt == 0 and v == 0 then
            stats.ff = stats.ff + 1
        end
    end
    return stats
end
stats = computeStatsFromState(state)

function love.load()
    saveState()
end

local function select10()
    local start = state.pointer
    images = {}
    for i = 1, 10 do
        if start + i - 1 > #allImages then
            break
        end
        images[#images + 1] = lg.newImage(allImages[start + i - 1])
    end
    if #images == 0 then
        state.finished = true
    end
end
select10()

local t = 0

function love.draw()
    if state.finished then
        lg.setFont(font)
        lg.setColor(1, 1, 1)
        lg.printf("You have finished the set!", 0, lg.getHeight() * 0.4, lg.getWidth(), "center")
        lg.printf("Accuracy: %.2f%%", 0, lg.getHeight() * 0.5, lg.getWidth(), "center", 0, 1, 1, 0, lg.getFont():getHeight())
        lg.printf("Real accuracy: %.2f%%", 0, lg.getHeight() * 0.5 + lg.getFont():getHeight(), lg.getWidth(), "center", 0, 1, 1, 0, lg.getFont():getHeight())
        lg.printf("Fake accuracy: %.2f%%", 0, lg.getHeight() * 0.5 + lg.getFont():getHeight() * 2, lg.getWidth(), "center", 0, 1, 1, 0, lg.getFont():getHeight())
        return
    end
    lg.setFont(fontSmall)
    lg.setColor(0, 0, 0, 0.2)
    local s = string.format("Decider #%d | Set progress %d/10", state.pointer, 10 - #images)
    lg.rectangle("fill", 7, 7, lg.getFont():getWidth(s) + 6, lg.getFont():getHeight() + 6, 2, 2)
    lg.setColor(1, 1, 1)
    lg.print(s, 10, 10)

    lg.setFont(font)
    lg.setColor(1, 1, 1)
    local w, h = lg.getWidth(), lg.getHeight()
    local image = images[currentImage]
    local iw, ih = image:getDimensions()
    local scale = math.min(w / iw, h / ih) * 0.8
    lg.draw(image, w / 2 + touchDelta.x, h / 2 + touchDelta.y, 0, scale, scale, iw / 2, ih / 2)
    if t < 3 then
        if t <= 2.5 then
            lg.setColor(1, 1, 1, math.min(0.5, t))
            lg.rectangle("fill", w * 0.05, h * 0.05, w * 0.4, h * 0.9, 8, 8)
            lg.rectangle("fill", w * 0.55, h * 0.05, w * 0.4, h * 0.9, 8, 8)
            lg.setColor(0, 0, 0, math.min(0.5, t))
            lg.printf("Real", w * 0.05, h * 0.5 - lg.getFont():getHeight() * 0.5, w * 0.4, "center")
            lg.printf("Fake", w * 0.55, h * 0.5 - lg.getFont():getHeight() * 0.5, w * 0.4, "center")
        else
            lg.setColor(1, 1, 1, math.max(0, 3 - t))
            lg.rectangle("fill", w * 0.05, h * 0.05, w * 0.4, h * 0.9, 8, 8)
            lg.rectangle("fill", w * 0.55, h * 0.05, w * 0.4, h * 0.9, 8, 8)
        end
    else
        if love.system.getOS() == "Android" or love.system.getOS() == "iOS" or DEBUG_ANDROID then
            lg.setColor(1, 1, 1, 0.3)
            lg.setFont(fontSmall)
            lg.printf("Swipe left or right to decide", 0, h * 0.05, w, "center")
            lg.setFont(font)
            if type(touchDelta.lockId) ~= "number" then
                local is, x, y = pcall(function() return love.touch.getPosition(touchDelta.lockId) end)
                if is then
                    if math.abs(touchDelta.x) > lg.getWidth() * 0.15 then
                        if touchDelta.x < -lg.getWidth() * 0.15 then
                            lg.setColor(1, 1, 1, 0.1)
                            lg.rectangle("fill", w * 0.02, h * 0.02, w * 0.47, h * 0.96, 8, 8)
                            lg.setColor(1, 1, 1, 0.6)
                            lg.printf("Real", w * 0.02, h * 0.5 - lg.getFont():getHeight() * 0.5, w * 0.47, "center")
                        elseif touchDelta.x > lg.getWidth() * 0.15 then
                            lg.setColor(1, 1, 1, 0.1)
                            lg.rectangle("fill", w * 0.51, h * 0.02, w * 0.47, h * 0.96, 8, 8)
                            lg.setColor(1, 1, 1, 0.6)
                            lg.printf("Fake", w * 0.51, h * 0.5 - lg.getFont():getHeight() * 0.5, w * 0.47, "center")
                        end
                    end
                end
            end
        else
            local mx, my = love.mouse.getPosition()
            if mx > w * 0.05 and mx < w * 0.45 and my > h * 0.05 and my < h * 0.95 then
                lg.setColor(1, 1, 1, 0.1)
                lg.rectangle("fill", w * 0.05, h * 0.05, w * 0.4, h * 0.9, 8, 8)
                lg.setColor(1, 1, 1, 0.6)
                if love.mouse.isDown(1) and selectOnRelease.left then
                    lg.printf("Real", w * 0.05, h * 0.5 - lg.getFont():getHeight() * 0.5 - 24, w * 0.4, "center")
                    lg.setFont(fontSmall)
                    lg.printf("Release to submit", w * 0.05, h * 0.5 + 24, w * 0.4, "center")
                else
                    lg.printf("Real", w * 0.05, h * 0.5 - lg.getFont():getHeight() * 0.5, w * 0.4, "center")
                end
            elseif mx > w * 0.55 and mx < w * 0.95 and my > h * 0.05 and my < h * 0.95 then
                lg.setColor(1, 1, 1, 0.1)
                lg.rectangle("fill", w * 0.55, h * 0.05, w * 0.4, h * 0.9, 8, 8)
                lg.setColor(1, 1, 1, 0.6)
                if love.mouse.isDown(1) and selectOnRelease.right then
                    lg.printf("Fake", w * 0.55, h * 0.5 - lg.getFont():getHeight() * 0.5 - 24, w * 0.4, "center")
                    lg.setFont(fontSmall)
                    lg.printf("Release to submit", w * 0.55, h * 0.5 + 24, w * 0.4, "center")
                else
                    lg.printf("Fake", w * 0.55, h * 0.5 - lg.getFont():getHeight() * 0.5, w * 0.4, "center")
                end
            end
        end
    end
    if accs and #accs > 0 then
        lg.setFont(fontSmall)
        local str = string.format("~ Set finished ~\nAccuracy: %.2f%%\nReal accuracy: %.2f%%\nFake accuracy: %.2f%%", accs[1] * 100, accs[2] * 100, accs[3] * 100)
        lg.setColor(0, 0, 0, 0.5)
        lg.rectangle("fill", w * 0.5 - lg.getFont():getWidth(str) * 0.5 - 30, h * 0.5 - lg.getFont():getHeight() * 2.2 - 10, lg.getFont():getWidth(str) + 60, lg.getFont():getHeight() * 4.4 + 20, 9, 9)
        lg.setColor(1, 1, 1)
        lg.printf(str, 0, h * 0.5 - lg.getFont():getHeight() * 2, w, "center")
        lg.setFont(font)
    end
end

local function showAcc(accuracy, realAccuracy, fakeAccuracy)
    accs = {accuracy, realAccuracy, fakeAccuracy, 3}
end


local function decide(decision)
    state.decisions[allImages[state.pointer + currentImage - 1]] = decision
    state.pointer = state.pointer + currentImage
    saveState()
    submit(allImages[state.pointer - 1], decision)
    -- Pop the image from images
    table.remove(images, currentImage)
    -- Contrib. to local stats
    local gt = getGT(allImages[state.pointer - 1])
    if gt == 1 and decision == 1 then
        stats.rr = stats.rr + 1
    elseif gt == 1 and decision == 0 then
        stats.rf = stats.rf + 1
    elseif gt == 0 and decision == 1 then
        stats.fr = stats.fr + 1
    elseif gt == 0 and decision == 0 then
        stats.ff = stats.ff + 1
    end
    if #images == 0 then
        -- Show the stats
        local accuracy = (stats.rr + stats.ff) / (stats.rr + stats.rf + stats.fr + stats.ff + 1e-6)
        local realAccuracy = stats.rr / (stats.rr + stats.rf + 1e-6)
        local fakeAccuracy = stats.ff / (stats.fr + stats.ff + 1e-6)
        showAcc(accuracy, realAccuracy, fakeAccuracy)
        select10()
    end
    love.window.setTitle(string.format("Decider #%d | Set progress %d/10", state.pointer, 10 - #images))
end

function love.touchpressed(id, x, y, dx, dy, pressure)
    touchDelta.lockId = id
end

function love.touchmoved(id, x, y, dx, dy, pressure)
    if touchDelta.lockId == id then
        touchDelta.x = touchDelta.x + dx
        touchDelta.y = touchDelta.y + dy
    end
end

function love.touchreleased(id, x, y, dx, dy, pressure)
    local margin = lg.getWidth() * 0.15
    if touchDelta.lockId == id then
        if touchDelta.x < -margin then
            decide(1)
            touchDelta.action = 'decideLeft'
        elseif touchDelta.x > margin then
            decide(0)
            touchDelta.action = 'decideRight'
        else
            touchDelta.lockId = -1
            touchDelta.lastReleaseT = t
            touchDelta.action = 'reset'
            touchDelta.srcX, touchDelta.srcY = touchDelta.x, touchDelta.y
        end
    end
end


function love.mousepressed(x, y, button)
    if love.system.getOS() == "Android" then return end
    if t >= 3 then
        if x > lg.getWidth() * 0.05 and x < lg.getWidth() * 0.45 and y > lg.getHeight() * 0.05 and y < lg.getHeight() * 0.95 then
            selectOnRelease.left = true
            selectOnRelease.right = false
        elseif x > lg.getWidth() * 0.55 and x < lg.getWidth() * 0.95 and y > lg.getHeight() * 0.05 and y < lg.getHeight() * 0.95 then
            selectOnRelease.left = false
            selectOnRelease.right = true
        else
            selectOnRelease.left = false
            selectOnRelease.right = false
        end
    end
    if DEBUG_ANDROID then
        love.touchpressed("external", x, y, 0, 0, 1)
    end
end

function love.mousemoved(x, y, dx, dy, istouch)
    if love.system.getOS() == "Android" then return end
    if DEBUG_ANDROID then
        love.touchmoved("external", x, y, dx, dy, 1)
    end
end

function love.mousereleased(x, y, button)
    if love.system.getOS() == "Android" then return end
    if t >= 3 then
        if x > lg.getWidth() * 0.05 and x < lg.getWidth() * 0.45 and y > lg.getHeight() * 0.05 and y < lg.getHeight() * 0.95 then
            if selectOnRelease.left then
                decide(1)
            end
        elseif x > lg.getWidth() * 0.55 and x < lg.getWidth() * 0.95 and y > lg.getHeight() * 0.05 and y < lg.getHeight() * 0.95 then
            if selectOnRelease.right then
                decide(0)
            end
        end
    end
    if DEBUG_ANDROID then
        love.touchreleased("external", x, y, 0, 0, 1)
    end
end

function love.keypressed(key)
    if key == "left" or key == "a" then
        decide(1)
    elseif key == "right" or key == "d" then
        decide(0)
    elseif key == "r" and love.keyboard.isDown("lctrl") and love.keyboard.isDown("lshift") and love.keyboard.isDown("lalt") then
        state.pointer = 1
        state.decisions = {}
        stats = {
            rr = 0,
            rf = 0,
            fr = 0,
            ff = 0
        }
        saveState()
        select10()
    end
end


function love.update(dt)
    t = t + dt
    if accs and #accs > 0 then
        accs[4] = accs[4] - dt
        if accs[4] <= 0 then
            accs = {}
        end
    end
    if touchDelta.action == 'reset' then
        touchDelta.x = easing.outCubic(t - touchDelta.lastReleaseT, touchDelta.srcX, -touchDelta.srcX, 0.5)
        touchDelta.y = easing.outCubic(t - touchDelta.lastReleaseT, touchDelta.srcY, - touchDelta.srcY, 0.5)
        if t - touchDelta.lastReleaseT >= 0.5 then
            touchDelta.action = nil
        end
    elseif touchDelta.action == 'decideLeft' then
        touchDelta.x = 0
        touchDelta.y = 0
        if t >= 0.5 then
            touchDelta.action = nil
        end
    elseif touchDelta.action == 'decideRight' then
        touchDelta.x = 0
        touchDelta.y = 0
        if t >= 0.5 then
            touchDelta.action = nil
        end
    end
end
