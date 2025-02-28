DECIDER_VERSION = "0.1.0"

local http = require("socket.http")
local url = require("socket.url")

local function urlEncode(list)
    local result = {}
    for k, v in pairs(list) do
        result[#result + 1] = url.escape(k) .. "=" .. url.escape(v)
    end
    return table.concat(result, "&")
end

local function getUsername()
    if love.filesystem.getInfo("username.txt") then
        return love.filesystem.read("username.txt")
    end
    local isSuccess, u = pcall(function()
        return io.popen("whoami"):read("*l")
    end)
    if isSuccess then
        love.filesystem.write("username.txt", u)
        return u
    else
        return "unknown"
    end
end

local function submit(imagePath, decision)
    -- To disable all submit actions, uncomment the following line.
    return
    -- local data = {
    --     image = imagePath,
    --     decision = decision,
    --     username = getUsername()
    -- }
    -- local postData = urlEncode(data)
    -- local baseUrl = "http://101.132.113.116:23343/api/v1/decider"

    -- local body, code, _ = http.request(baseUrl.."?"..postData)
    -- if code ~= 200 then
    --     love.window.showMessageBox("Error", "Failed to submit the decision.\nFile: " .. imagePath .. "\nDecision: " .. decision, "error")
    -- end
end

return submit
