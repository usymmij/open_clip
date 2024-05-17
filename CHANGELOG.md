# CHANGELOG
> This is a changelog for this fork, 
> listing changes made to the original open_clip repository, forked from commit 2e8de8312ea6185df7bc24a73b19a195a801a9fc

## List of Changes
- [access model from main()](access-model-from-main())
- [feature projection](feature-projection)
- [use feature projection](use-feature-projection)
# Changes

## commit 

### access-model-from-main()
> `src/training/main.py`
- training.main() now returns the model as its return value

### feature projection
> `src/open_clip/projection.py`
- created the FeatureToEmbedding class
 
### use feature projection
>`src/open_clip/transformer.py`
- import FeatureToEmbedding, and replace the projection in the VisionTransformer class


