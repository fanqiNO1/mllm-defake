function love.conf(t)

    t.highdpi = true
    t.identity = "Decider (Debug @Win)"
    t.appendidentity = true

    t.externalstorage = true

    -- handheld, non-landscape
    t.window.width = 900
    t.window.title = "Decider (Debug @Win)"
    t.window.height = 600
    t.window.fullscreen = false
    t.window.resizable = true

    t.modules.video = false
    t.modules.audio = false
    t.modules.sound = false
    t.modules.physics = false
    t.modules.joystick = false

end
