## Lessons learned making Fonts

It’s been a month I’ve worked on font generation. Here are my reflections. 
I’ve learned a lot. I built my first somewhat complex frontend. I built a non-throwaway pipeline for the font generation, handling dozens of edge cases. 

### The key technical issues remain
- Consistency → all letters are present in the same style
- Customizability → ability to change the style of certain / all letters
- Speed → iterating should feel fast and fun

### Other thing I’ve learned
- Qwen 2.5 VL is an amazing model
- Generating your own font from scratch is hard
- Design is ALL about the details

### What was hardest to build:
- Kerning
- Glyph alignment 
- Consistency of style when generating glyphs. 

The issue is it's not usable yet. It still doesn't have punctuation and some characters are missing. It not at a level of quality where a designer would use it, and not at the level of customizability where it would appeal to marketers or founders.

I could work on it further, but the number of edge cases and details are not my jam. I am a 80/20 kind of person. The last 20 percent are much more work and a lot more tedious, but definitely required for a good-looking font. 

My passion lies with tools and small functions.
I have two paths I can take here, towards the designer or the casual users. I am more attracted by the causal user in this moment. 
One usage that excites me personally is a tool call that fixes / updates the text on an image. I wonder whether I should try building that. 

I might open-source the code here, if anyone wants to use it.
In the mean time feel free to [try the tool here](https://ai-font-generator.vercel.app/), send a dm or pull request for feedback :)


