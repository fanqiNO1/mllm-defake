# `mllm-defake` Human Tests

*This project is a stand-alone sub-project of the `mllm-defake` repository.*

It is a [LOVE2D](https://love2d.org/) program that allows human testers to submit whether they believe an image is real or fake. The program is designed to be used in conjunction with the `mllm-defake` project to collect human feedback and compare it to other methods.

Researchers may find it useful to collect human feedback on a dataset of images. This program allows users to view images and submit whether they believe the image is real or fake. The program will then save the user's responses to a local file and upload them to a server.

To run this project, first install the multi-platform [LOVE2D engine](https://love2d.org/). After `love` is installed, run the following command:

```bash
love --console .
```

Where `.` is the directory that contains this file. The program will load the first image in the `images/` directory sequentially and display it. The user can then press `A` for real images and `D` for fake images. Arrow keys (left/right) also works.

> Mobile platforms are also supported with a swipe-based control. Swipe left for real images and swipe right for fake images.

The local save directory is `%APPDATA%/LOVE/decider/` on Windows. [This LOVE2D doc page](http://love2d.org/wiki/love.filesystem) shows where you can find the annotation file locally, on all supported platforms.
